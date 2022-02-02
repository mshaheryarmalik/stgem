#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time
from performance import PerformanceData

class Algorithm:
    """
    Base class for all test suite generation algorithms.
    """

    def __init__(self, sut, test_repository, objective_func, objective_selector, logger=None):
        self.sut = sut
        self.test_repository = test_repository
        self.objective_func = objective_func
        self.objective_selector = objective_selector

        self.logger = logger
        self.log = ( lambda s: self.logger.algorithm.info(s) if logger is not None else None)

        self.test_suite = []

        self.perf = PerformanceData()

    def __getattr__(self, name):
        value = self.parameters.get(name)
        if value is None:
            raise AttributeError(name)

        return value

    def generate_test(self):
        raise NotImplementedError()
