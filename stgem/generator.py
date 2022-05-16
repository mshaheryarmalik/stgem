import copy, os, time, datetime, random

from collections import namedtuple
from multiprocessing import Pool
import dill as pickle

import numpy as np
import torch

from stgem.algorithm.algorithm import Algorithm
from stgem.budget import Budget
from stgem.logger import Logger
from stgem.objective_selector import ObjectiveSelectorAll
from stgem.sut import SUT
from stgem.test_repository import TestRepository

class StepResult:

    def __init__(self, test_repository, success, parameters):
        self.timestamp = datetime.datetime.now()
        self.test_repository = test_repository
        self.success = success
        self.parameters = parameters
        self.algorithm_performance = None
        self.model_performance = None

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
    def restore_from_file(file_name):
        with open(file_name, "rb") as file:
            obj = pickle.load(file)
        return obj

    def dump_to_file(self, file_name):
        # first create a temporary file
        temp_file_name = "{}.tmp".format(file_name)
        with open(temp_file_name, "wb") as file:
            pickle.dump(self, file)
        # then we rename it to its final name
        os.replace(temp_file_name, file_name)

class Step:

    def run(self) -> StepResult:
        raise NotImplementedError

    def setup(self, sut, test_repository, objective_funcs, objective_selector, device, logger):
        pass

class Search(Step):
    """
    A search step.
    """

    def __init__(self, algorithm: Algorithm, budget_threshold, mode="exhaust_budget"):
        self.algorithm = algorithm
        self.budget = None
        self.budget_threshold = budget_threshold
        self.logger = None

        if mode not in ["exhaust_budget", "stop_at_first_objective"]:
            raise Exception("Unknown test generation mode '{}'.".format(mode))
        self.mode = mode

    def setup(self, sut, test_repository, budget, objective_funcs, objective_selector, device, logger):
        self.budget = budget
        self.logger = logger
        self.log = lambda msg: (self.logger("step", msg) if logger is not None else None)
        self.algorithm.setup(
            sut=sut,
            test_repository=test_repository,
            budget=budget,
            objective_funcs=objective_funcs,
            objective_selector=objective_selector,
            device=device,
            logger=logger)

    def run(self) -> StepResult:
        self.budget.update_threshold(self.budget_threshold)

        # allow the algorithm to initialize itself
        self.algorithm.initialize()

        if not (self.mode == "stop_at_first_objective" and self.algorithm.test_repository.minimum_objective == 0.0):
            success = False
            generator = self.algorithm.generate_test()

            # TODO: We should check if the budget was exhausted during the test
            # generation and discard the final test if this is so.
            i = 0
            while self.budget.remaining():
                try:
                    idx = next(generator)
                except StopIteration:
                    self.log("Generator finished before budget was exhausted.")
                    break

                if not success and np.min(self.algorithm.test_repository.minimum_objective) == 0:
                    self.log("First success at test {}.".format(i + 1))
                    success = True

                if success and self.mode == "stop_at_first_objective":
                    break

                i += 1
        else:
            success = True

        # Allow the algorithm to store trained models or other generated data.
        self.algorithm.finalize()

        # Report results.
        self.log("Step minimum objective component: {}".format(self.algorithm.test_repository.minimum_objective))

        # Save certain parameters in the StepResult object.
        parameters = {}
        parameters["algorithm_name"] = self.algorithm.__class__.__name__
        parameters["algorithm"] = copy.deepcopy(self.algorithm.parameters)
        if "device" in parameters["algorithm"]:
            del parameters["algorithm"]["device"]
        parameters["model_name"] = [self.algorithm.models[i].__class__.__name__ for i in range(self.algorithm.N_models)]
        parameters["model"] = [copy.deepcopy(self.algorithm.models[i].parameters) for i in range(self.algorithm.N_models)]
        parameters["objective_name"] = [objective.__class__.__name__ for objective in self.algorithm.objective_funcs]
        parameters["objective"] = [copy.deepcopy(objective.parameters) for objective in self.algorithm.objective_funcs]
        parameters["objective_selector_name"] = self.algorithm.objective_selector.__class__.__name__
        parameters["objective_selector"] = copy.deepcopy(self.algorithm.objective_selector.parameters)

        # Build the StepResult object.
        step_result = StepResult(self.algorithm.test_repository, success, parameters)
        step_result.algorithm_performance = self.algorithm.perf
        step_result.model_performance = [self.algorithm.models[i].perf for i in range(self.algorithm.N_models)]

        return step_result

class STGEM:

    def __init__(self, description, sut: SUT, budget: Budget, objectives, objective_selector=None, steps=[]):
        self.description = description
        self.sut = sut
        self.budget = budget
        self.objectives = objectives

        if objective_selector is None:
            objective_selector = ObjectiveSelectorAll()
        self.objective_selector = objective_selector

        self.steps = steps
        self.device = None

        self.logger = Logger()

    def setup_sut(self):
        self.sut.setup(self.budget, self.sut_rng)

    def setup_objectives(self):
        # Setup the objective functions for optimization.
        for o in self.objectives:
            o.setup(self.sut)

        # Setup the objective selector.
        self.objective_selector.setup(self.objectives)

    def setup_seed(self, seed=None):
        self.seed = seed
        # We use a random seed unless it is specified.
        # Notice that making Pytorch deterministic makes it a lot slower.
        if self.seed is not None:
            torch.use_deterministic_algorithms(mode=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            self.seed = random.randint(0, 2 ** 15)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # A random source for SUT for deterministic random samples from the
        # input space.
        self.sut_rng = np.random.RandomState(seed=self.seed)

    def setup_steps(self):
        for step in self.steps:
            step.setup(
                sut=self.sut,
                test_repository=self.test_repository,
                budget=self.budget,
                objective_funcs=self.objectives,
                objective_selector=self.objective_selector,
                device=self.device,
                logger=self.logger)

    def setup(self, seed=None):
        self.setup_seed(seed=seed)

        self.setup_sut()

        # Setup the device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup the test repository.
        self.test_repository = TestRepository()

        self.setup_objectives()

        self.setup_steps()

    def _run(self, silent=False) -> STGEMResult:
        # Running this assumes that setup has been run.
        results = []

        silent_tmp = self.logger.silent
        self.logger.silent = silent

        # Setup and run steps sequentially.
        step_results = []
        for step in self.steps:
            step_results.append(step.run())

        sr = STGEMResult(self.description, self.sut.__class__.__name__, copy.deepcopy(self.sut.parameters), self.seed, self.test_repository, step_results, self.sut.perf)

        self.logger.silent = silent_tmp

        return sr

    def run(self, seed=None, silent=False) -> STGEMResult:
        self.setup(seed)
        return self._run(silent=silent)

