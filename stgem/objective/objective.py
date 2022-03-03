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
    Objective function to falsify a STL specification.
    """

    def __init__(self, sut, specification):
        super().__init__(sut)

        # rtamt may have dependency problems. We continue even if we cannot import it
        try:
            import rtamt
        except:
            print("Cannot import rtamt. Objectives using rtamt will throw an exception.")
            import traceback
            traceback.print_exc()

        self.dim = 1
        self.specification = specification

        # Create the RTAMT specification.
        # We use discrete time STL
        self.spec = rtamt.STLSpecification()
        for var in self.sut.outputs:
            self.spec.declare_var(var, float)
        self.spec.spec = specification()

    def _evaluate_vector(self, output):
        pass

    def _evaluate_signal(self, timestamps, signals):
        # Find the step length of the timestamps (in seconds) and setup time
        # correctly. Currently we only support fixed step length.
        if len(timestamps) == 1:
            raise Exception("A signal should be defined on at least two time steps.")
        step = timestamps[1] - timestamps[0]
        for i in range(2, len(timestamps)):
            if timestamps[i] - timestamps[i-1] != step:
                raise Exception("Timestamps with variable step length not supported.")

        # Scale the signals.

        self.spec.set_sampling_period(step, "s", 0.1)

        # We need to parse only after setting the sampling period.
        try:
            self.spec.parse()
        except:
            # TODO: Handle errors.
            raise

        # Check that the horizon of the formula is equals the length of the
        # signal. If this is not the case, we would need more logic to extract
        # the correct robustness value.
        horizon = self.spec.top.horizon
        print("Horizon: {}".format(horizon))

        # Transform the STL formula to past temporal logic.
        try:
            self.spec.pastify()
        except:
            # TODO: Handle errors.
            raise

        trajectories = {"time": timestamps}
        for i, name in enumerate(self.sut.outputs):
            trajectories[name] = signals[i]

        # The final value is the correct one when we assume that the horizon
        # equals the signal length.
        robustness = self.spec.evaluate(trajectories)[-1][1]
        # Clip the robustness to [0, 1].
        robustness = max(0, min(robustness, 1))

        return robustness

    def __call__(self, *args, **kwargs):
        # If we have a single argument, then we treat it as a vector input.
        # Otherwise we assume that we have a signal input timestaps, signals.
        if len(args) == 1:
            self._evaluate_vector(args[0])
        else:
            self._evaluate_signal(timestamps=args[0], signals=args[1])

