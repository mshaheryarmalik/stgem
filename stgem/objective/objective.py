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

class Minimize(Objective):
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
            ranges = np.asarray([[-I[1], -I[0]] for I in self.sut.output_range[idx]])
        else:
            ranges = [self.sut.output_range[i] for i in idx]

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
        # For signals we use the dense time STL because it is unclear how to
        # use the discrete version. The horizons are especially unclear in the 
        # discrete case. For vector outputs we use discrete time STL because
        # updating with a single signal value does not seem to give sensible
        # results otherwise.
        self.spec_dense = rtamt.STLDenseTimeSpecification()
        self.spec_discrete = rtamt.STLSpecification()
        for var in self.sut.outputs:
            self.spec_dense.declare_var(var, "float")
            self.spec_discrete.declare_var(var, "float")
        self.spec_dense.spec = specification
        self.spec_discrete.spec = specification

    def _evaluate_vector(self, output):
        # We assume that the output is a single observation of a signal. It
        # follows that not all STL formulas have a clear interpretation (like
        # always[0,30](x1 > 0 and x2 > 0). It is up to the user to ensure a
        # reasonable interpretation.

        spec = self.spec_discrete

        # Scale the input.
        output = self.sut.scale(np.asarray(output).reshape(1, -1), self.sut.output_range, target_A=0, target_B=1).reshape(-1)

        spec.reset()

        # We need to parse only after setting the sampling period.
        try:
            spec.parse()
	    # Transform the STL formula to past temporal logic.
            spec.pastify()
        except:
            # TODO: Handle errors.
            raise

		# Use the online monitor to get the robustness. We evaluate at time 0.
        robustness = spec.update(0, zip(self.sut.outputs, output))

        # Clip the robustness to [0, 1].
        robustness = max(0, min(robustness, 1))

        return robustness

    def _evaluate_signal(self, timestamps, signals):
        # Here we find the robustness at time 0.
        #
        # We assume that the user guarantees that time is increasing and that
        # timestamps do not overlap. It's best to use timestamps where the
        # difference between consecutive times is approximately constant.

        spec = self.spec_dense

        if timestamps[0] != 0:
            raise Exception("The first timestamp should be 0.")

        # Scale the signals.
        signals = [self.sut.scale_signal(signals[i], self.sut.output_range[i], target_A=0, target_B=1) for i in range(len(signals))]

        spec.reset()

        try:
            spec.parse()
        except:
            # TODO: Handle errors.
            raise

        # This needs to be fetched at this point.
        horizon = spec.top.horizon

        if horizon > timestamps[-1]:
            raise Exception("The horizon of the formula is too long compared to input signal length. The robustness cannot be computed.")

        # Transform the STL formula to past temporal logic.
        try:
            spec.pastify()
        except:
            # TODO: Handle errors.
            raise

        trajectories = []
        for i, name in enumerate(self.sut.outputs):
            trajectory = [[timestamps[j], signals[i][j]] for j in range(len(timestamps))]
            trajectories.append([name, trajectory])

        robustness_signal = spec.evaluate(*trajectories)
        robustness = None
        for t, r in robustness_signal:
            if t == horizon:
                robustness = r
                break

        if robustness is None:
            raise Exception("Could not figure out correct robustness at horizon {} from robustness signal {}.".format(horizon, robustness_signal))

        # Clip the robustness to [0, 1].
        robustness = max(0, min(robustness, 1))

        return robustness

    def __call__(self, *args, **kwargs):
        # If we have a single argument, then we treat it as a vector input.
        # Otherwise we assume that we have a signal input timestaps, signals.
        if len(args) == 1:
            return self._evaluate_vector(args[0])
        else:
            return self._evaluate_signal(timestamps=args[0], signals=args[1])

