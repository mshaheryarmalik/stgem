#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import copy
from stgem.performance import PerformanceData

class SearchSpace:
    @property
    def input_dimensions(self):
        raise NotImplementedError

    @property
    def output_dimensions(self):
        raise NotImplementedError

    def is_valid(self,test) -> bool:
        raise NotImplementedError

    def sample_input_space(self):
        raise NotImplementedError


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


    def setup(self,  search_space: SearchSpace , device=None, logger=None):

        self.search_space =  search_space
        print(self.search_space)
        self.device = device
        self.logger = logger
        self.log = (lambda s: self.logger.algorithm.info(s) if logger is not None else None)

        # Setup the device.
        self.parameters["device"] = device
        # Set input dimension.
        if not "input_dimension" in self.parameters:
            self.parameters["input_dimension"] = self.search_space.input_dimensions

        def copy_input_dimension(d, idim):
            for k,v in d.items():
                if v=="copy:input_dimension":
                   d[k] = idim
                if isinstance(v, dict):
                    copy_input_dimension(v,idim)

        copy_input_dimension(self.parameters, self.search_space.input_dimensions)

        # create models
        if self.model_factory:
            self.models = [self.model_factory() for _ in range(self.search_space.output_dimensions)]

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

    def train(self, active_outputs, test_repository):
        self.perf.timer_start("training")
        self.do_train(active_outputs,test_repository)
        self.perf.save_history("training_time", self.perf.timer_reset("training"))

    def do_train(self, active_outputs, test_repository):
        pass

    def generate_next_test(self):
        self.perf.timer_start("generation")
        r=self.do_generate_next_test()
        assert r
        self.perf.save_history("generation_time", self.perf.timer_reset("generation"))
        return r

    def do_generate_next_test(self):
       raise NotImplementedError

    def finalize(self):
        """A Step calls this method after the budget has been exhausted and the
        algorithm will no longer be used."""

        pass


