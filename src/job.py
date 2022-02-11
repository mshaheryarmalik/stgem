
import os, datetime, logging, random
from collections import namedtuple
import json

import torch
import numpy as np

import sut, objective, algorithm
from test_repository import TestRepository


class Job:
    def __init__(self, description=None):
        if description is None:
            self.description= {}
        else:
            self.description=description
            self.setup()

    def setup_from_file(self,file_name):
        with open(file_name) as f:
            self.description=json.load(f)
        self.setup()
        return self

    def setup(self):
        def dict_access(d, s):
            current = d
            for k in s.split("."):
                current = current[k]
            return current

        def dict_set(d, s, v):
            pcs = s.split(".")
            current = d
            for k in pcs[:-1]:
                current = current[k]
            current[pcs[-1]] = v

        # Implement the copying mechanism of values.
        keys = list(self.description.keys())
        for key in keys:
            item = dict_access(self.description, key)
            if isinstance(item, dict):
                keys += [key + "." + k for k in item.keys()]
            elif isinstance(item, str) and item.startswith("copy:"):
                dict_set(self.description, key, dict_access(self.description, item[5:]))

        # Fill in empty values for certain parameters if missing.
        for name in ["sut_parameters", "objective_selector_parameters"]:
            if not name in self.description:
                self.description[name] = {}
        for i in range(len(self.description["objective_func"]) - len(self.description["objective_func_parameters"])):
            self.description["objective_func_parameters"].append({})

        # Setup seed.
        # TODO: Make configurable.
        # Notice that making Pytorch deterministic makes it a lot slower.
        SEED = random.randint(0, 2**15)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(mode=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Setup the device.
        self.description["algorithm_parameters"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup loggers.
        logger_names = ["algorithm", "model"]
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
        loggers = {x: logging.getLogger(x) for x in ["algorithm", "model"]}
        for logger in loggers.values():
            logger.setLevel("DEBUG")
        logger = namedtuple("Logger", logger_names)(**loggers)

        # Setup the system under test.
        sut_class = sut.loadSUT(self.description["sut"])
        asut = sut_class(parameters=self.description.get("sut_parameters",{}))

        # Setup the test repository.
        test_repository = TestRepository()

        # Setup the objective functions for optimization.
        N_objectives = 0
        objective_funcs = []
        for n, s in enumerate(self.description["objective_func"]):
            objective_class = objective.loadObjective(s)
            objective_func = objective_class(sut=asut, **self.description["objective_func_parameters"][n])
            N_objectives += objective_func.dim
            objective_funcs.append(objective_func)

        # Setup the objective selector.
        objective_selector_class = objective.loadObjectiveSelector(self.description["objective_selector"])
        objective_selector = objective_selector_class(N_objectives=N_objectives, **self.description["objective_selector_parameters"])

        # Process job parameters for algorithm setup.
        # Setup the initial random tests to 20% unless the value is user-set.
        if not "N_random_init" in self.description["job_parameters"]:
            self.description["job_parameters"]["N_random_init"] = int(0.2 * self.description["job_parameters"]["N_tests"])

        # Select the algorithm to be used and setup it.
        # TODO: predefined random data loader
        self.description["algorithm_parameters"]["N_tests"] = self.description["job_parameters"]["N_tests"]
        self.description["algorithm_parameters"]["N_random_init"] = self.description["job_parameters"]["N_random_init"]
        algorithm_class = algorithm.loadAlgorithm(self.description["algorithm"])
        self.algorithm = algorithm_class(sut=asut,
                                         test_repository=test_repository,
                                         objective_funcs=objective_funcs,
                                         objective_selector=objective_selector,
                                         parameters=self.description["algorithm_parameters"],
                                         logger=logger
                                        )

        return self

    def start(self):

        mode = "exhaust_budget" if "mode" not in self.description["job_parameters"] else self.description["job_parameters"]["mode"]
        if mode not in ["exhaust_budget", "stop_at_first_falsification"]:
            raise Exception("Unknown test generation mode '{}'.".format(mode))

        falsified = False
        generator = self.algorithm.generate_test()
        outputs = []

        for i in range(self.description["job_parameters"]["N_tests"]):

            idx = next(generator)
            _, output = self.algorithm.test_repository.get(idx)
            outputs.append(output)

            if not falsified and np.min(output) == 0:
                print("First falsified at test {}.".format(i + 1))
                falsified = True
            if falsified and mode == "stop_at_first_falsification":
                break

        if falsified:
            return True
        else:
            print("Could not falsify within the given budget.")
            print("Minimum objective components:")
            print(np.min(np.asarray(outputs), axis=0))
            return False
