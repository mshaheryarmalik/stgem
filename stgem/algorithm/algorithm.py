#!/usr/bin/python3
# -*- coding: utf-8 -*-

import copy

from stgem.performance import PerformanceData

class Algorithm:
    """
    Base class for all test suite generation algorithms.
    """

    default_parameters = {}

    def __init__(self, model_factory=None, model=None, models=None, parameters=None):
        if sum([model is not None, models is not None, model_factory is not None])>1:
            raise TypeError("You can provide only one of these input parameters:  model_factory, model,  models")
        if not models:
            models=[]

        self.model= model
        self.models= models
        self.model_factory = model_factory
        self.search_space= None
        self.models = []
        self.perf=PerformanceData()

        if parameters is None:
            parameters = {}

        # merge deafult_parameters and parameters, the later takes priority if a key appears in both dictionaries
        # the result is a new dictionary
        self.parameters = self.default_parameters | parameters

    def setup(self, search_space, device=None, logger=None):
        self.search_space =  search_space
        self.device = device
        self.logger = logger
        self.log = lambda msg: (self.logger("algorithm", msg) if logger is not None else None)

        # Set input dimension.
        if not "input_dimension" in self.parameters:
            self.parameters["input_dimension"] = self.search_space.input_dimension

        # Create models by cloning
        if self.model:
            self.models=[]
            for _ in range(self.search_space.output_dimension):
                self.models.append(copy.deepcopy(self.model))

        # Create models by factory
        if self.model_factory:
            self.models = [self.model_factory() for _ in range(self.search_space.output_dimension)]

        # setup the models
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


