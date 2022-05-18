#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import copy

from stgem.performance import PerformanceData

class SearchSpace:

    def __init__(self):
        self.sut = None
        self.rng = None

    def setup(self, sut, rng):
        self.sut = sut
        self.rng = rng

    @property
    def input_dimension(self):
        return self.sut.idim

    @property
    def output_dimension(self):
        return self.sut.odim

    def is_valid(self, test) -> bool:
        return self.sut.validity(test)

    def sample_input_space(self):
        return self.rng.uniform(-1, 1, size=self.input_dimension)

class Algorithm:
    """
    Base class for all test suite generation algorithms.
    """

    default_parameters = {}

    def __init__(self, model_factory=None, parameters=None):
        self.model_factory = model_factory
        self.search_space= None
        self.models = []
        self.perf=PerformanceData()

        if parameters is None:
            parameters = copy.deepcopy(self.default_parameters)

        self.parameters = parameters

    def setup(self, search_space: SearchSpace, device=None, logger=None):
        self.search_space =  search_space
        self.device = device
        self.logger = logger
        self.log = lambda msg: (self.logger("algorithm", msg) if logger is not None else None)

        # Setup the device.
        self.parameters["device"] = device
        # Set input dimension.
        if not "input_dimension" in self.parameters:
            self.parameters["input_dimension"] = self.search_space.input_dimension

        # Create models.
        if self.model_factory:
            self.models = [self.model_factory() for _ in range(self.search_space.output_dimension)]
        self.N_models = len(self.models)

        for m in self.models:
            m.setup(self.search_space, self.device, self.logger)

    def __getattr__(self, name):
        if "parameters" in self.__dict__:
            if name in self.parameters:
                return self.parameters.get(name)

        raise AttributeError(name)

    def initialize(self):
        """A Step calls this method before the first generate_test call"""
        pass

    def train(self, active_outputs, test_repository, budget_remaining):
        self.perf.timer_start("training")
        self.do_train(active_outputs, test_repository, budget_remaining)
        self.perf.save_history("training_time", self.perf.timer_reset("training"))

    def do_train(self, active_outputs, test_repository, budget_remaining):
        raise NotImplementedError

    def generate_next_test(self, active_outputs, test_repository, budget_remaining):
        self.perf.timer_start("generation")
        r = self.do_generate_next_test(active_outputs, test_repository, budget_remaining)
        self.perf.save_history("generation_time", self.perf.timer_reset("generation"))

        return r

    def do_generate_next_test(self, active_outputs, test_repository, budget_remaining):
       raise NotImplementedError

    def finalize(self):
        """A Step calls this method after the budget has been exhausted and the
        algorithm will no longer be used."""

        pass


