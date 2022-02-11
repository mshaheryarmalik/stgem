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

import numpy as np

from performance import PerformanceData

class SUT:
    """
    Base class implementing a system under test.
    """

    def __init__(self):
        self.perf = PerformanceData()
        # The variables below are set by the inheriting classes.
        #
        # The input dimension of the SUT (number of components for
        # vector-valued inputs and number of signals for signal-valued inputs).
        self.idim = None
        # The output dimension of the SUT (number of components for
        # vector-valued outputs and number of signals for signal-valued
        # outputs).
        self.odim = None
        # We always assume that inputs are scaled to [-1, 1], so a range for
        # inputs must be specified (a list of 2-tuples representing intervals).
        # For example, self.irange = [(0, 2), (-15, 15)] would indicate range
        # [0, 2] for the first component and (-15, 15) for the second.
        #
        # IMPORTANT: The values self.irange and self.orange must be numpy
        # arrays! The above notation is just for clarity.
        #
        # Outputs are not scaled to [-1, 1] by default, but this can be
        # achieved by specifying an output range and by using the self.scale
        # method. As the output range is unknown, the value None can be used to
        # indicate this. For example, self.orange = [(-300, 100), None]
        # specifies output range [-300, 100] for the first component and the
        # range of the second component is unknown.
        #
        # NOTICE: While we support different ranges for each output value, they
        # should in fact be the same. This is because most algorithms we use
        # directly compare the objective function values in [0, 1], so they
        # should in some sense be comparable. This is approximately achieved by
        # having the same range and by clipping to [0, 1].
        self.irange = None
        self.orange = None

    def scale(self, x, intervals, target_A=-1, target_B=-1):
        """
        Return a scaled x where the components of x with the specified
        intervals are scaled to the interval [A, B] (default [-1, 1]).
        """

        y = np.zeros_like(x)
        for i in range(x.shape[1]):
            A = intervals[i][0]
            B = intervals[i][1]
            C = (target_B-target_A)/(B-A)
            D = target_A - C*A
            y[:,i] = C*x[:,i] + D

        return y

    def descale(self, x, intervals, A=-1, B=1):
        """
        Return a scaled x where the components of x in [A, B] (default [-1, 1])
        are scaled to the given intervals.
        """

        y = np.zeros_like(x)
        for i in range(x.shape[1]):
            target_A = intervals[i][0]
            target_B = intervals[i][1]
            C = (target_B-target_A)/(B-A)
            D = target_A - C*A
            y[:,i] = C*x[:,i] + D

        return y

    def execute_test(self, test):
        self.perf.timer_start("execution")
        r = self._execute_test(test)
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

    def min_distance(self, tests, x):
        """
        Returns the minimum distance of the given tests to the specified test.
        """

        return self._min_distance(tests, x)

