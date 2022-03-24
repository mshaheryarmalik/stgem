from collections import namedtuple
from typing import List
import dill as pickle
import torch
import os, time, datetime, random, logging
import numpy as np
from stgem.objective_selector import ObjectiveSelectorAll


from stgem.algorithm.algorithm import Algorithm
from stgem.sut import SUT

from stgem.test_repository import TestRepository


class StepResult:
    def __init__(self, description, test_repository, success):
        self.timestamp = datetime.datetime.now()
        self.description = description
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
    "A Search step"

    def __init__(self, algorithm: Algorithm, max_tests=0, max_time=0, mode="exhaust_budget"):
        self.algorithm = algorithm
        self.max_tests = max_tests
        self.max_time = max_time
        if mode not in ["exhaust_budget", "stop_at_first_objective"]:
            raise Exception("Unknown test generation mode '{}'.".format(mode))

        self.mode = mode

    def setup(self, sut, test_repository, objective_funcs, objective_selector, device, logger):
        self.algorithm.setup(
            sut=sut,
            test_repository=test_repository,
            objective_funcs=objective_funcs,
            objective_selector=objective_selector,
            max_steps=self.max_tests,
            device=device,
            logger=logger)

    def run(self) -> StepResult:

        if self.max_time == 0 and self.max_tests == 0:
            raise Exception("Step description does not specify neither a maximum time nor a maximum number tests.")

        # allow the algorithm to initialize itself
        self.algorithm.initialize()

        success = False
        generator = self.algorithm.generate_test()
        outputs = []

        i = 0
        start_time = time.perf_counter()
        elapsed_time = 0

        while (self.max_tests == 0 or i < self.max_tests) and (self.max_time == 0 or elapsed_time < self.max_time):
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
            elapsed_time = time.perf_counter() - start_time

        # allow the algorithm to store trained models or other generated data
        self.algorithm.finalize()

        # report resuts
        if len(outputs) > 0:
            print("Step  minimum objective components:")
            print(np.min(np.asarray(outputs), axis=0))

        step_result = StepResult(self, self.algorithm.test_repository, success)
        step_result.algorithm_performance = self.algorithm.perf
        step_result.model_performance = [self.algorithm.models[i].perf for i in range(self.algorithm.N_models)]

        return step_result

class STGEM:
    def __init__(self, description, sut: SUT, objectives, objective_selector=None, steps=[]):
        self.description = description
        self.sut = sut
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

    def setup_objectives(self):
        # Setup the objective functions for optimization.
        for o in self.objectives:
            o.setup(self.sut)

        # Setup the objective selector.
        self.objective_selector.setup(self.objectives)


    def setup_seed(self):
        # Setup seed.
        # ---------------------------------------------------------------------
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

    def setup(self):

        self.setup_seed()
        # Run secondary initializer.
        self.sut.initialize()

        # Setup the device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup the test repository.
        self.test_repository = TestRepository()

        self.setup_objectives()

    def run(self, seed=None) -> STGEMResult:

        self.seed = seed
        self.setup()

        results = []

        for step in self.steps:
            step.setup(
                sut=self.sut,
                test_repository=self.test_repository,
                objective_funcs=self.objectives,
                objective_selector=self.objective_selector,
                device=self.device,
                logger=self.logger)
            results.append(step.run())

        sr = STGEMResult(self.description, self.test_repository, results, self.sut.perf)

        return sr
