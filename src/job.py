
import os, datetime, logging
from collections import namedtuple

import torch
import numpy as np

import sut, objective, algorithm
from test_repository import TestRepository


class Job:
    def __init__(self, description):
        self.description=description
        self.setup()

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
        asut = sut_class(**self.description["sut_parameters"])

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
        objective_selector = objective_selector_class(N_objectives=N_objectives)

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

    def start(self):
        generator = self.algorithm.generate_test()
        for i in range(self.description["job_parameters"]["N_tests"]):
            next(generator)


