#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, datetime, logging
from collections import namedtuple

import torch

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

        keys = list(self.description.keys())
        for key in keys:
            item = dict_access(self.description, key)
            if isinstance(item, dict):
                keys += [key + "." + k for k in item.keys()]
            elif isinstance(item, str) and item.startswith("copy:"):
                dict_set(self.description, key, dict_access(self.description, item[5:]))

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
        objective_class = objective.loadObjective(self.description["objective_func"])
        objective_func = objective_class(**self.description["objective_func_parameters"])
        target = None

        # Setup the objective selector.
        objective_class = objective.loadObjectiveSelector(self.description["objective_selector"])
        if not "objective_selector_parameters" in self.description:
            self.description["objective_selector_parameters"] = {}
        objective_selector = objective_class(objective_func=objective_func, **self.description["objective_selector_parameters"])

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
                                    objective_func=objective_func,
                                    objective_selector=objective_selector,
                                    parameters=self.description["algorithm_parameters"],
                                    logger=logger)

    def start(self):
        generator = self.algorithm.generate_test()
        for i in range(self.description["job_parameters"]["N_tests"]):
            next(generator)


if __name__ == "__main__":
    # Random
    job_desc = {"sut": "odroid.OdroidSUT",
           "sut_parameters": {"data_file": "../data/odroid/odroid.npy"},
           "objective_func": "ObjectiveMaxSelected",
           "objective_func_parameters": {"selected": [0]},
           "objective_selector": "ObjectiveSelectorMAB",
           "objective_selector_parameters": {"warm_up": 60},
           "algorithm": "random.Random",
           "algorithm_parameters": {"use_predefined_random_data": False,
                                    "predefined_random_data": {"test_inputs": None,
                                                               "test_outputs": None}},
           "job_parameters": {"N_tests": 300}
           }

    # Setup the experiment.
    ajob = Job(job_desc)

    # Start the job.
    ajob.start()
