#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import copy
from stgem.performance import PerformanceData


class Algorithm:
    """
    Base class for all test suite generation algorithms.
    """

    default_parameters={}

    def __init__(self, model_factory=None, parameters=None):

        self.model_factory=model_factory
        self.N_models = 0
        self.models = []

        if parameters is None:
            parameters = copy.copy(self.default_parameters)

        self.parameters=parameters
        self.perf = PerformanceData()

    def create_models(self):
        if self.model_factory:
            self.N_models = sum(1 for f in self.objective_funcs)
            self.models = [ self.model_factory()  for _ in range(self.N_models)]
        else:
            self.N_models=0
            self.models=[]

    def setup(self, sut, test_repository, objective_funcs, objective_selector, max_steps, device=None, logger=None):

        self.sut = sut
        self.test_repository = test_repository
        self.objective_funcs = objective_funcs
        self.objective_selector = objective_selector
        self.device=device
        self.logger = logger
        self.log = (lambda s: self.logger.algorithm.info(s) if logger is not None else None)

        # Setup the device.
        self.parameters["device"] = device
        # Set input dimension, max_tests and preceding_tests in parameters
        if not "input_dimension" in self.parameters:
            self.parameters["input_dimension"] = self.sut.idim


        self.parameters["max_tests"] = max_steps
        if not "preceding_tests" in self.parameters:
            self.parameters["preceding_tests"] = self.test_repository.tests

        def copy_input_dimension(d, idim):
            for k,v in d.items():
                if v=="copy:input_dimension":
                   d[k] = idim
                if isinstance(v, dict):
                    copy_input_dimension(v,idim)

        copy_input_dimension(self.parameters, self.sut.idim)
        self.create_models()

        for m in self.models:
            m.setup(self.sut, self.logger)

    def __getattr__(self, name):
        if "parameters" in self.__dict__:
            if name in self.parameters:
                return self.parameters.get(name)

        raise AttributeError(name)



    def initialize(self):
        """
           A Step calls this method before the first generate_test call
        """
        pass

    def generate_test(self):
        raise NotImplementedError()

    def finalize(self):
        """
        A Step calls this method after all tests have been generated and the algorithm
        will not be used anymore in that step.
        """
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
