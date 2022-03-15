#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import numpy as np

from stgem.performance import PerformanceData

class Algorithm:
    """
    Base class for all test suite generation algorithms.
    """

    def __init__(self, sut, test_repository, objective_funcs, objective_selector, parameters, logger=None):
        self.sut = sut
        self.test_repository = test_repository
        self.objective_funcs = objective_funcs
        self.objective_selector = objective_selector
        self.parameters = parameters

        self.logger = logger
        self.log = (lambda s: self.logger.algorithm.info(s) if logger is not None else None)

        self.models=[]
        self.N_models=0

        self.perf = PerformanceData()

    def __getattr__(self, name):
        value = self.parameters.get(name)
        if value is None:
            raise AttributeError(name)

        return value

    def generate_test(self):
        raise NotImplementedError()

class LoaderAlgorithm(Algorithm):
    """
    Algorithm which simply loads pregenerated data from a file.
    """

    # TODO: Currently this is a placeholder and does nothing.

    def __init__(self, sut, test_repository, objective_funcs, objective_selector, parameters, logger=None):
        super().__init__(sut, test_repository, objective_funcs, objective_selector, parameters, logger)

        return
        # Check if the data file exists.
        if not os.path.exists(self.data_file):
            raise Exception("Pregenerated date file '{}' does not exist.".format(self.data_file))

    def generate_test(self):
        return
        yield

