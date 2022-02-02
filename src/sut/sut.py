#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Here is a base class implementation for systems under test (SUTs). We do not
strictly enforce the input and output representations for flexibility, but we
have the following conventions which should be followed if possible.

Inputs:
-------
We have two input formats: vectors and discrete signals.

Vectors inputs should be numpy arrays of floats. The SUT should allow the
execution of variable-length input vectors whenever this makes sense (e.g.,
when the input is interpretable as time series).

Discrete signals.
"""
from performance import PerformanceData

class SUT:
    """
    Base class implementing a system under test.
    """
    def __init__(self):
        self.perf= PerformanceData()

    def execute_test(self, test):
        self.perf.timer_start("execution")
        r= self._execute_test(test)
        self.perf.save_history("execution_time", self.perf.timer_reset("execution"))
        return r

    def _execute_test(self, test):
        raise NotImplementedError()

    def execute_random_test(self):
        raise NotImplementedError()

    def sample_input_space(self):
        raise NotImplementedError()

    def validity(self, test):
        """
        Basic validator which deems all tests valid.
        """

        return 1

