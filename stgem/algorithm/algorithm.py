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
        self.n_inputs = 0
        self.n_outputs = 0
        self.models = []

        if parameters is None:
            parameters = copy.deepcopy(self.default_parameters)

        self.parameters = parameters

    def create_models(self):
        if self.model_factory:
            self.models = [self.model_factory() for _ in range(self.n_outputs)]

    def setup(self, n_inputs, n_outputs, device=None, logger=None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.device = device
        self.logger = logger
        self.log = (lambda s: self.logger.algorithm.info(s) if logger is not None else None)

        # Setup the device.
        self.parameters["device"] = device
        # Set input dimension.
        if not "input_dimension" in self.parameters:
            self.parameters["input_dimension"] = self.n_inputs

        def copy_input_dimension(d, idim):
            for k,v in d.items():
                if v=="copy:input_dimension":
                   d[k] = idim
                if isinstance(v, dict):
                    copy_input_dimension(v,idim)

        copy_input_dimension(self.parameters, self.sut.idim)
        self.create_models()

        for m in self.models:
            m.setup(self.sut, self.device, self.logger)

    def __getattr__(self, name):
        if "parameters" in self.__dict__:
            if name in self.parameters:
                return self.parameters.get(name)

        raise AttributeError(name)

    def initialize(self):
        """A Step calls this method before the first generate_test call"""

        pass

    def generate_next_test(self):
       raise NotImplementedError

    def finalize(self):
        """A Step calls this method after the budget has been exhausted and the
        algorithm will no longer be used."""

        pass


