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
from collections import namedtuple
from stgem.performance import PerformanceData
from stgem.algorithm.algorithm import SearchSpace

# TODO Document this
# TODO Consider a dataclass

SUTResult = namedtuple("SUTResult", "inputs outputs input_timestamps output_timestamps error")

class SUT:
    """Base class implementing a system under test. """

    def __init__(self, parameters=None):
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters

        self.perf = PerformanceData()

        """
        All SUTs have the below variables which concern inputs and outputs,
        their ranges etc. We describe them here, but do not set them. They are
        set by generator.py and setting defaults for them here would complicate
        that code. They can be set (and in some cases should be) by inheriting
        classes.

        self.idim
        The input dimension of the SUT (number of components for vector-valued
        inputs and number of signals for signal-valued inputs).

        self.odim
        The output dimension of the SUT (number of components for vector-valued
        outputs and number of signals for signal-valued outputs).

        Names for inputs and outputs (strings).
        self.inputs
        self.outputs

        We always assume that inputs are scaled to [-1, 1], so a range for
        inputs must be specified (a list of 2-element lists representing
        intervals). For example, self.input_range = [[0, 2], [-15, 15]] would
        indicate range [0, 2] for the first component and [-15, 15] for the
        second.
        
        Outputs are not scaled to [-1, 1] by default, but this can be achieved
        by specifying an output range and by using the self.scale or
        self.scale_signal methods in _execute_test. If the output range is
        unknown, the value None can be used to indicate this. For example,
        self.output_range = [[-300, 100], None] specifies output range [-300,
        100] for the first component and the range of the second component is
        unknown.
        
        NOTICE: We support different ranges for each output value, but doing so
        is not always a good idea. This is because most algorithms we use
        directly compare the objective function values in [0, 1], so they
        should in some sense be comparable.
        """

    def __getattr__(self, name):
        if "parameters" in self.__dict__:
            if name in self.parameters:
                return self.parameters.get(name)

        raise AttributeError(name)

    def setup(self):
        """Setup the budget and perform steps necessary for two-step
        initialization. Derived classes should always call this super class
        setup method."""

        # Infer dimensions and names for inputs and outputs from impartial
        # information.

        # If self.inputs exists and is an integer, transform it into default
        # input names i1, ...iN where N is this integer. This also determines
        # idim if unset.
        if hasattr(self, "inputs") and isinstance(self.inputs, int):
            if not hasattr(self, "idim"):
                self.idim = self.inputs
            self.inputs = ["i{}".format(i) for i in range(self.inputs)]

        # If idim is not set, it can be inferred from input names (a list of
        # names) or input ranges.
        if hasattr(self, "idim"):
            # idim set already, set default input names if necessary.
            if not hasattr(self, "inputs"):
                self.inputs = ["i{}".format(i) for i in range(self.idim)]
        else:
            # idim can be inferred from input names, if defined.
            if hasattr(self, "inputs"):
                self.idim = len(self.inputs)
            else:
                # idim can be inferred from input ranges. Otherwise we do not
                # know what to do.
                if not hasattr(self, "input_range"):
                    raise Exception("SUT input dimension not defined and cannot be inferred.")
                self.idim = len(self.input_range)
                self.inputs = ["i{}".format(i) for i in range(self.idim)]

        # The same as above for outputs.
        if hasattr(self, "outputs") and isinstance(self.outputs, int):
            if not hasattr(self, "odim"):
                self.odim = self.outputs
            self.outputs = ["o{}".format(i) for i in range(self.outputs)]

        if hasattr(self, "odim"):
            if not hasattr(self, "outputs"):
                self.outputs = ["o{}".format(i) for i in range(self.odim)]
        else:
            if hasattr(self, "outputs"):
                self.odim = len(self.outputs)
            else:
                if not hasattr(self, "output_range"):
                    raise Exception("SUT output dimension not defined and cannot be inferred.")
                self.odim = len(self.output_range)
                self.outputs = ["o{}".format(i) for i in range(self.odim)]

        # Setup input and output ranges and fill unspecified input and output
        # ranges with Nones.
        if not hasattr(self, "input_range"):
            self.input_range = []
        if not isinstance(self.input_range, list):
            raise Exception("The input_range attribute of the SUT must be a Python list.")
        self.input_range += [None for _ in range(self.idim - len(self.input_range))]
        if not hasattr(self, "output_range"):
            self.output_range = []
        if not isinstance(self.output_range, list):
            raise Exception("The output attribute of the SUT must be a Python list.")
        self.output_range += [None for _ in range(self.odim - len(self.output_range))]

    def variable_range(self, var_name):
        """Return the range for the given variable (input or output)."""

        # NOTICE: Attributes might not exist unless the setup method has been called.
        if hasattr(self, "output_range"):
            for n, v in enumerate(self.outputs):
                if var_name == v:
                    return self.output_range[n]
        if hasattr(self, "input_range"):
            for  n, v in enumerate(self.inputs):
                if var_name == v:
                    return self.input_range[n]

        raise Exception("No variable '{}'.".format(var_name))

    def scale(self, x, intervals, target_A=-1, target_B=1):
        """
        Return a scaled x where the components of x with the specified
        intervals are scaled to the interval [A, B] (default [-1, 1]). If an
        interval is None, then no scaling is done.
        """

        if len(intervals) < x.shape[1]:
            raise Exception("Not enough intervals ({}) for scaling a vector of length {}.".format(len(intervals), x.shape[1]))

        y = np.zeros_like(x)
        for i in range(x.shape[1]):
            if intervals[i] is not None:
                A = intervals[i][0]
                B = intervals[i][1]
                C = (target_B-target_A)/(B-A)
                D = target_A - C*A
                y[:,i] = C*x[:,i] + D
            else:
                y[:,i] = x[:,i]

        return y

    def scale_signal(self, signal, interval, target_A=-1, target_B=1):
        """
        Scales the input signal whose values are in the given interval to the
        specified interval [A, B] (default [-1, 1]). If the interval is None,
        then no scaling is done.
        """

        y = []
        for v in signal:
            if interval is not None:
                A = interval[0]
                B = interval[1]
                C = (target_B-target_A)/(B-A)
                D = target_A - C*A
                y.append(C*v + D)
            else:
                y.append(v)

        return y

    def descale(self, x, intervals, A=-1, B=1):
        """
        Return a scaled x where the components of x in [A, B] (default [-1, 1])
        are scaled to the given intervals. If an interval is None, then no
        scaling is done.
        """

        if len(intervals) < x.shape[1]:
            raise Exception("Not enough intervals ({}) for descaling a vector of length {}.".format(len(intervals), x.shape[1]))

        y = np.zeros_like(x)
        for i in range(x.shape[1]):
            if intervals[i] is not None:
                target_A = intervals[i][0]
                target_B = intervals[i][1]
                C = (target_B-target_A)/(B-A)
                D = target_A - C*A
                y[:,i] = C*x[:,i] + D
            else:
                y[:,i] = x[:,i]

        return y

    def denormalize_test(self,test):
        return self.descale(test.reshape(1, -1), self.input_range).reshape(-1)

    def execute_test(self, test) -> SUTResult:
        raise NotImplementedError()

    def validity(self, test):
        """
        Basic validator which deems all tests valid.
        """

        return 1

