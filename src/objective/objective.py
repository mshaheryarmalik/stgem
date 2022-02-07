#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

"""
REMEMBER: Always clip the objective function values to [0, 1]. Otherwise we
can get very wild losses when training neural networks. This is not good as
then the loss minimization might focus on the wrong thing.
"""

class Objective:

    def __init__(self, sut):
        self.sut = sut

    def __call__(self, output):
        return output

    def _scale(self, x, intervals):
        """
        Scale the components of x in the given intervals to [0, 1].
        """

        for i in range(x.shape[1]):
            A = intervals[i][0]
            B = intervals[i][1]
            C = 1/(B-A)
            D = -A/(B-A)
            x[:,i] = C*x[:,i] + D

        return x

class ObjectiveMaxSelected(Objective):
    """
    Objective function for SUT with fixed-length vector outputs which selects
    the maximum among the specified components.
    """

    def __init__(self, sut, selected=None, scale=False, invert=False):
        super().__init__(sut)
        if not (isinstance(selected, list) or isinstance(selected, tuple) or selected is None):
            raise Exception("The parameter 'selected' must be None or a list or a tuple.")

        self.dim = 1
        self.selected = selected
        self.scale = scale
        self.invert = invert

    def __call__(self, output):
        if self.selected is None:
            idx = list(range(len(output)))
        else:
            idx = self.selected

        if self.invert:
            output = output*(-1)
            ranges = np.asarray([[-I[1], -I[0]] for I in self.sut.orange[idx]])
        else:
            ranges = self.sut.orange[idx]

        if self.scale:
            output = self.sut.scale(output[idx].reshape(1, -1), ranges, target_A=0, target_B=1).reshape(-1)
        else:
            output = output[idx]

        return max(0, min(1, max(output)))

class ObjectiveMaxComponentwise(Objective):
    """
    Objective function for SUT with signal outputs which computes the
    componentwise maxima of the signal.
    """

    def __init__(self):
        self.dim = 1

    def __call__(self, timestamps, signals):
        return [max(signal) for signal in signals]
