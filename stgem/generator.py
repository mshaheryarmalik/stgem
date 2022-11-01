import copy, datetime, gzip, os, random

import torch

import dill as pickle
import numpy as np

from stgem.algorithm.algorithm import Algorithm
from stgem.budget import Budget
from stgem.exceptions import *
from stgem.logger import Logger
from stgem.objective_selector import ObjectiveSelectorAll
from stgem.sut import SearchSpace, SUT, SUTInput
from stgem.test_repository import TestRepository

class StepResult:

    def __init__(self, test_repository, success, parameters):
        self.timestamp = datetime.datetime.now()
        self.test_repository = test_repository
        self.success = success
        self.parameters = parameters
        self.algorithm_performance = None
        self.model_performance = None
        self.models  = []
        self.final_model = None

class STGEMResult:

    def __init__(self, description, sut_name, sut_parameters, seed, test_repository, step_results, sut_performance):
        self.timestamp = datetime.datetime.now()
        self.description = description
        self.sut_name = sut_name
        self.sut_parameters = sut_parameters
        self.seed = seed
        self.step_results = step_results
        self.test_repository = test_repository
        self.sut_performance = sut_performance

    @staticmethod
    def restore_from_file(file_name: str):
        o = gzip.open if file_name.endswith(".gz") else open
        with o(file_name, "rb") as file:
            obj = pickle.load(file)
        return obj

    def dump_to_file(self, file_name: str):
        if os.path.exists(file_name):
            raise FileExistsError(file_name)

        o = gzip.open if file_name.endswith(".gz") else open
        # first create a temporary file
        temp_file_name = "{}.tmp".format(file_name)
        with o(temp_file_name, "wb") as file:
            pickle.dump(self, file)
        # then we rename it to its final name
        os.replace(temp_file_name, file_name)

class Step:

    def run(self) -> StepResult:
        raise NotImplementedError

    def setup(self, sut, search_space, test_repository, budget, objective_funcs, objective_selector, device, logger):
        self.sut = sut
        self.search_space = search_space
        self.test_repository = test_repository
        self.budget = budget
        self.objective_funcs = objective_funcs
        self.objective_selector = objective_selector
        self.device = device
        self.logger = logger
        self.log = lambda msg: (self.logger("step", msg) if logger is not None else None)

class Search(Step):
    """A search step."""

    def __init__(self, algorithm: Algorithm, budget_threshold, mode="exhaust_budget", results_include_models=False, results_checkpoint_period=1):
        self.algorithm = algorithm
        self.budget = None
        self.budget_threshold = budget_threshold
        if mode not in ["exhaust_budget", "stop_at_first_objective"]:
            raise Exception("Unknown test generation mode '{}'.".format(mode))
        self.mode = mode

        self.results_include_models = results_include_models
        self.results_checkpoint_period = results_checkpoint_period

    def setup(self, sut, search_space, test_repository, budget, objective_funcs, objective_selector, device, logger):
        super().setup(sut, search_space, test_repository, budget, objective_funcs, objective_selector, device, logger)

        self.algorithm.setup(
            search_space=self.search_space,
            device=self.device,
            logger=self.logger)

    def run(self) -> StepResult:
        self.budget.update_threshold(self.budget_threshold)

        # A list for saving model skeletons.
        model_skeletons = []

        # Allow the algorithm to initialize itself.
        self.algorithm.initialize()

        self.success = True
        if not (self.mode == "stop_at_first_objective" and self.test_repository.minimum_objective <= 0.0):
            self.success = False

            # TODO: We should check if the budget was exhausted during the test
            # generation and discard the final test if this is so.
            i = 0
            while self.budget.remaining() > 0:
                self.algorithm.train(self.objective_selector.select(), self.test_repository, self.budget.remaining())
                self.budget.consume("training_time", self.algorithm.perf.get_history("training_time")[-1])
                if not self.budget.remaining() > 0: break

                # TODO: Should we catch any exceptions here?
                self.log("Starting to generate test {}.".format(self.test_repository.tests + 1))
                try:
                    next_test = self.algorithm.generate_next_test(self.objective_selector.select(), self.test_repository, self.budget.remaining())
                except AlgorithmException:
                    # We encountered an algorithm error. There might be many
                    # reasons such as explosion of gradients. We take this as
                    # an indication that the algorithm is unable to keep going,
                    # so we exit.
                    break

                self.budget.consume("generation_time", self.algorithm.perf.get_history("generation_time")[-1])
                self.log("Generated test {}.".format(next_test))
                if not self.budget.remaining() > 0: break

                self.log("Executing the test...")
                self.algorithm.perf.timer_start("execution")
                sut_input = SUTInput(next_test, None, None)
                sut_result = self.sut.execute_test(sut_input)
                self.algorithm.perf.save_history("execution_time", self.algorithm.perf.timer_reset("execution"))
                self.budget.consume("executions")
                self.budget.consume("execution_time", self.algorithm.perf.get_history("execution_time")[-1])
                self.log("Input to the SUT: {}".format(sut_input))
                self.log("Result from the SUT: {}".format(sut_result))
                output = [objective(sut_input, sut_result) for objective in self.objective_funcs]
                self.log("The actual objective: {}".format(output))

                # TODO: Argmin does not take different scales into account.
                self.objective_selector.update(np.argmin(output))
                self.test_repository.record(sut_input, sut_result, output)

                if not self.success and self.test_repository.minimum_objective <= 0.0:
                    self.log("First success at test {}.".format(i + 1))
                    self.success = True

                # Save the models if requested.
                if self.results_include_models and self.results_checkpoint_period != 0 and i % self.results_checkpoint_period == 0:
                    model_skeletons.append([model.skeletonize() for model in self.algorithm.models])
                else:
                    model_skeletons.append(None)

                i += 1

                if self.success and self.mode == "stop_at_first_objective":
                    break

        # Allow the algorithm to store trained models or other generated data.
        self.algorithm.finalize()

        # Report results.
        self.log("Step minimum objective component: {}".format(self.test_repository.minimum_objective))

        result = self._generate_step_result(model_skeletons)

        return result

    def _generate_step_result(self, model_skeletons):
        # Save certain parameters in the StepResult object.
        parameters = {}
        parameters["algorithm_name"] = self.algorithm.__class__.__name__
        parameters["algorithm"] = copy.deepcopy(self.algorithm.parameters)
        parameters["model_name"] = [self.algorithm.models[i].__class__.__name__ for i in range(self.algorithm.N_models)]
        parameters["model"] = [copy.deepcopy(self.algorithm.models[i].parameters) for i in range(self.algorithm.N_models)]
        parameters["objective_name"] = [objective.__class__.__name__ for objective in self.objective_funcs]
        parameters["objective"] = [copy.deepcopy(objective.parameters) for objective in self.objective_funcs]
        parameters["objective_selector_name"] = self.objective_selector.__class__.__name__
        parameters["objective_selector"] = copy.deepcopy(self.objective_selector.parameters)

        # Build the StepResult object.
        step_result = StepResult(self.test_repository, self.success, parameters)
        step_result.algorithm_performance = self.algorithm.perf
        step_result.model_performance = [self.algorithm.models[i].perf for i in range(self.algorithm.N_models)]
        if self.results_include_models:
            step_result.models = model_skeletons
            step_result.final_models = [model.skeletonize() for model in self.algorithm.models]

        return step_result

class Load(Step):
    """Step which simply loads pregenerated data from a file."""

    def __init__(self, file_name, mode="initial", range_load=None):
        self.file_name = file_name
        if not os.path.exists(self.file_name):
            raise Exception("Pregenerated date file '{}' does not exist.".format(self.file_name))
        if mode not in ["initial", "random"]:
            raise ValueError("Unknown load mode '{}'.".format(mode))
        if range_load < 0:
            raise ValueError("The load range {} cannot be negative.".format(range_load))
        self.mode = mode
        self.range_load = range_load

    def run(self, results_include_models=False, results_checkpoint_period=1) -> StepResult:
        try:
            raw_data = STGEMResult.restore_from_file(self.file_name)
        except:
            raise Exception("Error loading STGEMResult object from file '{}'.".format(self.file_name))

        range_max = raw_data.test_repository.tests
        if self.range_load is None:
            self.range_load = range_max
        elif self.range_load > range_max:
            raise ValueError("The load range {} is out of bounds. Loaded maximum range for loaded data is {}.".format(self.range_load, range_max))

        if self.mode == "random":
            # Use the search space RNG to ensure consistent selection.
            idx = self.search_space.rng.choice(np.arange(range_max), size=self.range_load, replace=False)
        elif self.mode == "initial":
            idx = range(self.range_load)

        already_successful = self.test_repository.minimum_objective <= 0

        for i in idx:
            if self.budget.remaining() == 0: break
            x, y, z = raw_data.test_repository.get(i)

            if len(x.inputs) != self.search_space.input_dimension:
                raise ValueError("Loaded sample input dimension {} does not match SUT input dimension {}".format(len(x.inputs), self.search_space.input_dimension))
            if y.output_timestamps is None:
                if len(y.outputs) != self.search_space.output_dimension:
                    raise ValueError("Loaded sample vector output dimension {} does not match SUT vector output dimension {}.".format(len(y.outputs), self.search_space.output_dimension))
            else:
                if y.outputs.shape[0] != self.search_space.output_dimension:
                    raise ValueError("Loaded sample signal number {} does not match SUT signal number {}.".format(y.outputs.shape[0], self.search_space.output_dimension))

            self.test_repository.record(x, y, z)

        success = not already_successful and self.test_repository.minimum_objective <= 0

        # Save certain parameters in the StepResult object.
        parameters = {}
        parameters["file_name"] = self.file_name
        parameters["mode"] = self.mode
        parameters["load_range"] = self.range_load

        # Build StepResult object with test_repository
        step_result = StepResult(self.test_repository, success, parameters)

        return step_result

class STGEM:

    def __init__(self, description, sut: SUT, objectives, objective_selector=None, budget: Budget = None, steps=None):
        self.description = description
        # The description might be used as a file name, so we check for some
        # nongood characters.
        # TODO: Is this complete enough?
        nonsafe_chars = "/\<>:\"|?*"
        for c in self.description:
            if c in nonsafe_chars:
                raise ValueError("Character '{}' not allowed in a description (could be used as a file name).".format(c))
        self.sut = sut
        self.step_results = []

        if budget is None:
            budget = Budget()
        self.budget = budget

        self.objectives = objectives
        if objective_selector is None:
            objective_selector = ObjectiveSelectorAll()
        self.objective_selector = objective_selector

        self.steps = [] if steps is None else steps
        self.device = None

        self.logger = Logger()
        self.log = lambda msg: (self.logger("stgem", msg) if self.logger is not None else None)

    def setup_seed(self, seed=None):
        self.seed = seed
        # We use a random seed unless it is specified.
        # Notice that making Pytorch deterministic makes it a lot slower.
        if self.seed is not None:
            torch.use_deterministic_algorithms(mode=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            self.seed = random.randint(0, 2**15)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # A random source for SUT for deterministic random samples from the
        # input space.
        self.search_space_rng = np.random.RandomState(seed=self.seed)

    def setup_sut(self):
        self.sut.setup()

    def setup_search_space(self):
        self.search_space = SearchSpace()
        self.search_space.setup(sut=self.sut, objectives=self.objectives, rng=self.search_space_rng)

    def setup_objectives(self):
        for o in self.objectives:
            o.setup(self.sut)

        self.objective_selector.setup(self.objectives)

    def setup_steps(self):
        for step in self.steps:
            step.setup(
                sut=self.sut,
                search_space=self.search_space,
                test_repository=self.test_repository,
                budget=self.budget,
                objective_funcs=self.objectives,
                objective_selector=self.objective_selector,
                device=self.device,
                logger=self.logger)

    def setup(self, seed=None, use_gpu=True):
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type != "cuda":
                self.log("Warning: requested torch device 'cuda' but got '{}'.".format(self.device.type))
        else:
            self.device = torch.device("cpu")

        self.test_repository = TestRepository()

        self.setup_seed(seed=seed)
        self.setup_sut()
        self.setup_search_space()
        self.setup_objectives()
        self.setup_steps()

    def _generate_result(self, step_results):
        return STGEMResult(self.description, self.sut.__class__.__name__, copy.deepcopy(self.sut.parameters), self.seed, self.test_repository, step_results, self.sut.perf)

    def _run(self) -> STGEMResult:
        # Running this assumes that setup has been run.

        # Setup and run steps sequentially.
        self.step_results = []
        for step in self.steps:
            self.step_results.append(step.run())

        return self._generate_result(self.step_results)

    def run(self, seed=None) -> STGEMResult:
        self.setup(seed)
        return self._run()

