#!/usr/bin/python3
# -*- coding: utf-8 -*-


class Objective:
    pass


class ObjectiveMaxSelected(Objective):
    """
    Objective function for SUT with fixed-length vector outputs which selects the
    maximum among the specified components.
    """

    def __init__(self, selected=None):
        if not (isinstance(selected, list) or isinstance(selected, tuple) or selected is None):
            raise Exception("The parameter 'selected' must be None or a list or a tuple.")

        self.dim = 1
        self.selected = selected

    def __call__(self, output):
        if self.selected is None:
            idx = range(len(output))
        else:
            idx = self.selected
        return [max(output[i] for i in idx)]


class ObjectiveMaxComponentwise(Objective):
    """
    Objective function for SUT with signal outputs which computes the
    componentwise maxima of the signal.
    """

    def __init__(self):
        self.dim = 1

    def __call__(self, timestamps, signals):
        return [max(signal) for signal in signals]
