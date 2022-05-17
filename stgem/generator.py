import copy, os, time, datetime, random, logging

from collections import namedtuple
from multiprocessing import Pool
import dill as pickle

import numpy as np
import torch

from stgem.algorithm.algorithm import Algorithm, SearchSpace
from stgem.objective_selector import ObjectiveSelectorAll
from stgem.sut import SUT
from stgem.test_repository import TestRepository
from stgem.budget import Budget


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
        if mode not in ["exhaust_budget", "stop_at_first_objective"]:
            raise Exception("Unknown test generation mode '{}'.".format(mode))
        self.mode = mode

    def setup(self, sut, search_space, test_repository, budget, objective_funcs, objective_selector, device, logger):
        self.sut = sut
        self.search_space = search_space
        self.test_repository = test_repository
        self.budget = budget
        self.budget.update_threshold(self.budget_threshold)
        self.objective_funcs = objective_funcs
        self.objective_selector = objective_selector
        self.algorithm.setup(
            search_space=self.search_space,
            device=device,
            logger=logger)
        self.logger = logger
        self.log = (lambda s: self.logger.algorithm.info(s) if logger is not None else None)


    def run(self) -> StepResult:
        # allow the algorithm to initialize itself
        self.algorithm.initialize()

        if not (self.mode == "stop_at_first_objective" and self.test_repository.minimum_normalized_output == 0.0):
            success = False

            # TODO: We should check if the budget was exhausted during the test
            # generation and discard the final test if this is so.
            i = 0
            while self.budget.remaining() > 0:
                self.algorithm.train(self.objective_selector.select(), self.test_repository)

                try:
                    next_test = self.algorithm.generate_next_test(self.objective_selector.select(), self.test_repository)
                except StopIteration:
                    print("Generator finished before budget was exhausted.")
                    break

                # Consume generation budget.
                self.budget.consume("generation_time", self.algorithm.perf.get_history("generation_time")[-1] +
                                    self.algorithm.perf.get_history("training_time")[-1])

                self.log("Executing the test...")
                self.algorithm.perf.timer_start("execution")
                sut_result = self.sut.execute_test(next_test)
                self.algorithm.perf.save_history("execution_time", self.algorithm.perf.timer_reset("execution"))
                self.budget.consume("executions")
                self.budget.consume("execution_time", self.algorithm.perf.get_history("execution_time")[-1])
                self.log("Result from the SUT {}".format(sut_result))
                output = [objective(sut_result) for objective in self.objective_funcs]
                self.log("The actual objective {} for the generated test.".format(output))

                self.objective_selector.update(np.argmin(output))
                self.test_repository.record(self.sut.denormalize_test(next_test) , next_test,sut_result,output)

                if not success and np.min(self.test_repository.minimum_normalized_output) == 0:
                    print("First success at test {}.".format(i + 1))
                    success = True

                if success and self.mode == "stop_at_first_objective":
                    break

                i += 1
        else:
            success = True

        # allow the algorithm to store trained models or other generated data
        self.algorithm.finalize()

        # report resuts
        print("Step minimum objective component: {}".format(self.test_repository.minimum_normalized_output))

        # Save certain parameters in the StepResult object.
        parameters = {}
        parameters["algorithm_name"] = self.algorithm.__class__.__name__
        parameters["algorithm"] = copy.deepcopy(self.algorithm.parameters)
        if "device" in parameters["algorithm"]:
            del parameters["algorithm"]["device"]
        parameters["model_name"] = [self.algorithm.models[i].__class__.__name__ for i in range(self.algorithm.N_models)]
        parameters["model"] = [copy.deepcopy(self.algorithm.models[i].parameters) for i in range(self.algorithm.N_models)]
        parameters["objective_name"] = [objective.__class__.__name__ for objective in self.objective_funcs]
        parameters["objective"] = [copy.deepcopy(objective.parameters) for objective in self.objective_funcs]
        parameters["objective_selector_name"] = self.objective_selector.__class__.__name__
        parameters["objective_selector"] = copy.deepcopy(self.objective_selector.parameters)

        # Build the StepResult object.
        step_result = StepResult(self.test_repository, success, parameters)
        step_result.algorithm_performance = self.algorithm.perf
        step_result.model_performance = [self.algorithm.models[i].perf for i in range(self.algorithm.N_models)]

        return step_result

class LoaderStep(Step):
    """
    Step which simply loads pregenerated data from a file.
    """

    # TODO: Currently this is a placeholder and does nothing.

    def __init__(self, parameters=None):
        super().__init__(parameters)
        return
        # Check if the data file exists.
        if not os.path.exists(self.data_file):
            raise Exception("Pregenerated date file '{}' does not exist.".format(self.data_file))



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

        # Setup loggers.
        # ---------------------------------------------------------------------
        logger_names = ["algorithm", "model"]
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
        loggers = {x: logging.getLogger(x) for x in ["algorithm", "model"]}
        for logger in loggers.values():
            logger.setLevel("DEBUG")
        self.logger = namedtuple("Logger", logger_names)(**loggers)

    def setup_sut(self):
        self.sut.setup()

    def setup_search_space(self):
        self.search_space = SearchSpace()
        self.search_space.setup(sut=self.sut, rng=self.search_space_rng)

    def setup_objectives(self):
        # Setup the objective functions for optimization.
        for o in self.objectives:
            o.setup(self.sut)

        # Setup the objective selector.
        self.objective_selector.setup(self.objectives)

    def setup_seed(self):
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
        self.search_space_rng = np.random.RandomState(seed=self.seed)

    def setup(self):
        self.setup_seed()

        self.setup_sut()

        self.setup_search_space()

        # Setup the device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup the test repository.
        self.test_repository = TestRepository()

        self.setup_objectives()

    def run(self, seed=None) -> STGEMResult:
        self.seed = seed
        self.setup()

        # Setup and run steps sequentially.
        step_results = []
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
            step_results.append(step.run())

        sr = STGEMResult(self.description, self.sut.__class__.__name__, copy.deepcopy(self.sut.parameters), self.seed, self.test_repository, step_results, self.sut.perf)


        return sr

def run_one_job(parameters):
    r = parameters[0].run(seed=parameters[1])

    return r

def run_multiple_generators(N, description, seed_factory, sut_factory, budget_factory, objective_factory, objective_selector_factory, step_factory, callback=None):
    """Creates and runs multiple STGEM objects in parallel and collects
    information on each run."""

    # Create the STGEM objects to be run.
    jobs = []
    for i in range(N):
        x = STGEM(description=description,
                  sut=sut_factory(),
                  budget=budget_factory(),
                  objectives=objective_factory(),
                  objective_selector=objective_selector_factory(),
                  steps=step_factory()
                 )
        seed = seed_factory()
        jobs.append((x, seed))

    # Execute the generators.
    # We remove the jobs from the list in order to free resources.
    results = []
    while len(jobs) > 0:
        job = jobs.pop(0)
        results.append(run_one_job(job))

        if not callback is None:
            callback(results[-1])

    return results

    """
    # Matlab does not work with pickling used by multiprocessing.
    # See https://stackoverflow.com/questions/55885741/how-to-run-matlab-function-with-spark
    N_workers = 2
    with Pool(N_workers) as pool:
        r = pool.map(run_one_job, jobs)
        pool.close()
        pool.join()

    return r
    """

