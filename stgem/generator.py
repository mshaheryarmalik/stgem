import os, time, datetime, random, logging

from collections import namedtuple
import dill as pickle

import numpy as np
import torch

from stgem.algorithm.algorithm import Algorithm
from stgem.objective_selector import ObjectiveSelectorAll
from stgem.sut import SUT
from stgem.test_repository import TestRepository
from stgem.budget import Budget

class StepResult:

    def __init__(self, test_repository, success):
        self.timestamp = datetime.datetime.now()
        self.test_repository = test_repository
        self.success = success
        self.algorithm_performance = None
        self.model_performance = None

class STGEMResult:

    def __init__(self, description, test_repository, step_results, sut_performance):
        self.timestamp = datetime.datetime.now()
        self.description = description
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

    def setup(self, sut, test_repository, budget, objective_funcs, objective_selector, device, logger):
        self.budget = budget
        self.budget.update_threshold(self.budget_threshold)
        self.algorithm.setup(
            sut=sut,
            test_repository=test_repository,
            budget=budget,
            objective_funcs=objective_funcs,
            objective_selector=objective_selector,
            device=device,
            logger=logger)

    def run(self) -> StepResult:
        # allow the algorithm to initialize itself
        self.algorithm.initialize()

        success = False
        generator = self.algorithm.generate_test()
        outputs = []

        # TODO: We should check if the budget was exhausted during the test
        # generation and discard the final test if this is so.
        i = 0
        while self.budget.remaining() > 0:
            try:
                idx = next(generator)
            except StopIteration:
                print("Generator finished before budget was exhausted.")
                break
            _, output = self.algorithm.test_repository.get(idx)
            outputs.append(output)

            if not success and np.min(output) == 0:
                print("First success at test {}.".format(i + 1))
                success = True

            if success and self.mode == "stop_at_first_objective":
                break

            i += 1

        # allow the algorithm to store trained models or other generated data
        self.algorithm.finalize()

        # report resuts
        if len(outputs) > 0:
            print("Step minimum objective components:")
            print(np.min(np.asarray(outputs), axis=0))

        step_result = StepResult(self.algorithm.test_repository, success)
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

        # Setup loggers.
        # ---------------------------------------------------------------------
        logger_names = ["algorithm", "model"]
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
        loggers = {x: logging.getLogger(x) for x in ["algorithm", "model"]}
        for logger in loggers.values():
            logger.setLevel("DEBUG")
        self.logger = namedtuple("Logger", logger_names)(**loggers)

    def setup_sut(self):
        self.sut.setup(self.budget)

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
        if self.seed:
            torch.use_deterministic_algorithms(mode=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            self.seed = random.randint(0, 2 ** 15)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def setup(self, seed=None):
        self.setup_seed(seed)

        self.setup_sut()

        # Setup the device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup the test repository.
        self.test_repository = TestRepository()

        self.setup_objectives()

    def _run(self) -> STGEMResult:
        # Running this assumes that setup has been run.
        results = []

        # Setup and run steps sequentially.
        for step in self.steps:
            step.setup(
                sut=self.sut,
                test_repository=self.test_repository,
                budget=self.budget,
                objective_funcs=self.objectives,
                objective_selector=self.objective_selector,
                device=self.device,
                logger=self.logger)
            results.append(step.run())

        sr = STGEMResult(self.description, self.test_repository, results, self.sut.perf)

        return sr

    def run(self, seed=None) -> STGEMResult:
        self.setup(seed)
        return self._run()

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

