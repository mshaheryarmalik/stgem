#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import copy
from stgem.performance import PerformanceData

class Algorithm:
    """
    Base class for all test suite generation algorithms.
    """

    default_parameters = {}

    def __init__(self, model_factory=None, parameters=None):
        self.model_factory = model_factory
        self.N_models = 0
        self.models = []

        if parameters is None:
            parameters = copy.deepcopy(self.default_parameters)

        self.parameters = parameters
        self.perf = PerformanceData()

    def setup(self, sut, test_repository, budget, objective_funcs, objective_selector, device=None, logger=None):
        self.sut = sut
        self.test_repository = test_repository
        self.budget = budget
        self.objective_funcs = objective_funcs
        self.objective_selector = objective_selector
        self.device = device
        self.logger = logger
        self.log = lambda msg: (self.logger("algorithm", msg) if logger is not None else None)

        # Setup the device.
        self.parameters["device"] = device
        # Set input dimension.
        if not "input_dimension" in self.parameters:
            self.parameters["input_dimension"] = self.sut.idim

        # Create models.
        if self.model_factory:
            self.N_models = sum(1 for f in self.objective_funcs)
            for _ in range(self.N_models):
                model = self.model_factory()
                model.setup(self.sut, self.device, self.logger)
                self.models.append(model)

    def __getattr__(self, name):
        if "parameters" in self.__dict__:
            if name in self.parameters:
                return self.parameters.get(name)

        raise AttributeError(name)

    def initialize(self):
        """A Step calls this method before the first generate_test call"""

        pass

    def generate_test(self):
        raise NotImplementedError()

    def finalize(self):
        """A Step calls this method after the budget has been exhausted and the
        algorithm will no longer be used."""

        pass

class LoaderAlgorithm(Algorithm):
    """
    Algorithm which simply loads pregenerated data from a file.
    """

    # TODO: Currently this is a placeholder and does nothing.

    def __init__(self, parameters=None):
        super().__init__(parameters)
        return
        # Check if the data file exists.
        if not os.path.exists(self.data_file):
            raise Exception("Pregenerated date file '{}' does not exist.".format(self.data_file))

    def generate_test(self):
        return
        yield
