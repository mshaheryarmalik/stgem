#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

# rtamt may have dependency problems. We continue even if we cannot import it
try:
    import rtamt
except:
    print("Cannot import rtamt. Objectives using rtamt will throw an exception.")
    import traceback
    traceback.print_exc()


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



class ObjectiveMinSelected(Objective):
    """
    Objective function for a SUT with fixed-length vector outputs which selects
    the minimum among the specified components.
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
            ranges = [self.sut.orange[i] for i in idx]

        if self.scale:
            output = self.sut.scale(output[idx].reshape(1, -1), ranges, target_A=0, target_B=1).reshape(-1)
        else:
            output = output[idx]

        return max(0, min(1, min(output)))

class ObjectiveMinComponentwise(Objective):
    """
    Objective function for a SUT with signal outputs which outputs the minima
    of the signals.
    """

    def __init__(self):
        self.dim = 1

    def __call__(self, timestamps, signals):
        return [min(signal) for signal in signals]

class FalsifySTL(Objective):
    """
    Objective function to falsify a STL specification
    """
    def __init__(self, sut, specification):
        super().__init__(sut)
        self.dim = 1
        self.specification=specification
        if not "outputs" in sut.parameters:
            raise Exception("SUS should have a sut_parameter named outputs containing a list of strings with the names of its outputs. ")

    def __call__(self, output):
        # This code only works for outputs as vectors
        # TODO: Extend this for signal outputs

        # 1. Create the RTAMT spect
        # We recreate the spec objects at every iteration, if not it uses the previous values
        self.spec = rtamt.STLSpecification()
        for var in self.sut.parameters["outputs"]:
            self.spec.declare_var(var, 'float')
        self.spec.spec = self.specification
        self.spec.parse()
        self.spec.pastify()

        # 2. Scale output, so we get a robutness between [0,1]
        ranges = self.sut.orange
        output = self.sut.scale(output.reshape(1, -1), ranges, target_A=0, target_B=1).reshape(-1)

        # 3. Get robustness
        rob= self.spec.update(0, zip(self.sut.parameters["outputs"],output ))

        # 4. Clip robustness in [0,1]
        rob=max(0,min(rob,1))

        return rob
