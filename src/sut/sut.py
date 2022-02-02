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


class SUT:
    """
    Base class implementing a system under test.
    """

    def execute_test(self, test):
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
