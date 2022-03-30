#!/usr/bin/python3
# -*- coding: utf-8 -*-

from itertools import chain

import numpy as np

from stgem.sut import SUTResult

"""
REMEMBER: Always clip the objective function values to [0, 1]. Otherwise we
can get very wild losses when training neural networks. This is not good as
then the loss minimization might focus on the wrong thing.
"""

class Objective:

    def setup(self, sut):
        self.sut = sut

    def __call__(self, r: SUTResult):
        return r.outputs

class Minimize(Objective):
    """
    Objective function for a SUT with fixed-length vector outputs which selects
    the minimum among the specified components.
    """

    def __init__(self, selected=None, scale=False, invert=False):
        super().__init__()
        if not (isinstance(selected, list) or isinstance(selected, tuple) or selected is None):
            raise Exception("The parameter 'selected' must be None or a list or a tuple.")

        self.dim = 1
        self.selected = selected
        self.scale = scale
        self.invert = invert

    def __call__(self, r: SUTResult):
        assert r.output_timestamps is None

        if self.selected is None:
            idx = list(range(len(r.outputs)))
        else:
            idx = self.selected

        if self.invert:
            outputs = r.outputs*(-1)
            ranges = np.asarray([[-self.sut.output_range[i][1], -self.sut.output_range[i][0]] for i in idx])
        else:
            ranges = [self.sut.output_range[i] for i in idx]

        if self.scale:
            output = self.sut.scale(r.outputs[idx].reshape(1, -1), ranges, target_A=0, target_B=1).reshape(-1)
        else:
            output = r.outputs[idx]

        return max(0, min(1, min(output)))

class ObjectiveMinComponentwise(Objective):
    """
    Objective function for a SUT with signal outputs which outputs the minima
    of the signals.
    """

    def __init__(self):
        super().__init__()
        self.dim = 1

    def __call__(self, r: SUTResult):
        assert r.output_timestamps is not None
        return [min(output) for output in r.outputs]

class FalsifySTL(Objective):
    """
    Objective function to falsify a STL specification.
    """

    def __init__(self, specification, strict_horizon_check=True):
        super().__init__()

        self.dim = 1
        self.specification = specification
        self.strict_horizon_check = strict_horizon_check

    def setup(self, sut):
        # Create the RTAMT specification.
        # We use discrete time for both vectors and signals. For vector outputs
        # discrete time needs to be used as with dense time updating with a
        # single signal value does not seem to give sensible values.
        super().setup(sut)

        # rtamt may have dependency problems. We continue even if we cannot import it
        try:
            import rtamt
        except:
            print("Cannot import rtamt. Objectives using rtamt will throw an exception.")
            import traceback
            traceback.print_exc()

        self.spec = rtamt.STLSpecification()
        for var in chain(self.sut.outputs, self.sut.inputs):
            self.spec.declare_var(var, "float")
        self.spec.spec = self.specification

    def _evaluate_vector(self,  output, clip=True):
        # We assume that the output is a single observation of a signal. It
        # follows that not all STL formulas have a clear interpretation (like
        # always[0,30](x1 > 0 and x2 > 0). It is up to the user to ensure a
        # reasonable interpretation.

        # Scale the input.
        output = self.sut.scale(np.asarray(output).reshape(1, -1), self.sut.output_range, target_A=0, target_B=1).reshape(-1)

        self.spec.reset()

        # We need to parse only after setting the sampling period.
        try:
            self.spec.parse()
	    # Transform the STL formula to past temporal logic.
            self.spec.pastify()
        except:
            # TODO: Handle errors.
            raise

        # Use the online monitor to get the robustness. We evaluate at time 0.
        robustness = self.spec.update(0, zip(self.sut.outputs, output))

        # Clip the robustness to [0, 1].
        if clip:
            robustness = max(0, min(robustness, 1))

        return robustness

    def _evaluate_signal(self, result, clip=True):
        input_timestamps = result.input_timestamps
        output_timestamps = result.output_timestamps
        input_signals = result.inputs
        output_signals = result.outputs

        """
        Here we find the robustness at time 0.
        
        We assume that the user guarantees that time is increasing. It is very
        difficult to understand how RTAMT works. It seems that with discrete
        time the actual timestamps are mostly ignored (at least as of
        29.3.2022). At least if the first two observations of signals have,
        say, timestamps 0 and 0.1, when evaluate (offline) or update (online)
        is called, what is actually done is effectively the same as if the
        timestamps were 0 and 1. The computed robustness signal uses the
        original timestamps, but they are not processed in any way. Both
        evaluate and update check the difference between the timestamps and
        increment the violation counter if the difference is too small compared
        to the sampling period (set by calling spec.set_sampling_period). Thus
        it seems that setting the sampling period is unnecessary. Setting it
        has the nasty consequence that then the formula time horizon is
        scaled according to the sampling period. Moreover it seems that this
        scaling works incorrectly. For example with a formula always[0,20] of
        time horizon 20 and sampling period 0.01, the horizon is set to 200000
        whereas it should be 2000 (it seems that the correct answer 2000 is
        incorrectly divided by the sampling period 0.01). Thus we skip setting
        the sampling period.
        """

        self.spec.reset()

        try:
            self.spec.parse()
        except:
            # TODO: Handle errors.
            raise

        # Find out which variables are in the STL formula. Adjust input and
        # output signals to have common timestamps if required.
        # ---------------------------------------------------------------------
        formula_variables = self.spec.top.out_vars.copy()
        # Separate to input and output variables and create a mapping for easy
        # access to correct signal.
        input_var = []
        output_var = []
        M = {}
        for var in formula_variables:
            try:
                M[var] = self.sut.outputs.index(var)
                output_var.append(var)
            except ValueError:
                try:
                    M[var] = self.sut.inputs.index(var)
                    input_var.append(var)
                except ValueError:
                    raise Exception("Variable '{}' not in input or output variables.".format(var))

        # These two checks make the adjustment code valid.
        if output_timestamps[0] != 0 or (input_timestamps is not None and input_timestamps[0] != 0):
            raise Exception("The first timestamp should be 0 in both input and output signals.")
        if input_timestamps is not None and input_timestamps[-1] != output_timestamps[-1]:
            raise Exception("The final timestamp should be equal for both input and output.")

        if len(input_var) > 0 and len(output_var) > 0:
            # Adjust signals.
            timestamps = []
            signals = {var:[] for var in formula_variables}
            i = 0; j = 0
            eps = 1e-6
            while i < len(input_timestamps) and j < len(output_timestamps):
                if abs(input_timestamps[i] - output_timestamps[j]) < eps:
                    # Same timestamp in both.
                    timestamps.append(output_timestamps[j])
                    # Use input as is.
                    for var in input_var:
                        signals[var].append(input_signals[M[var]][i])
                    # Use output as is.
                    for var in output_var:
                        signals[var].append(output_signals[M[var]][j])
                    i += 1; j += 1
                elif input_timestamps[i] - output_timestamps[j] >= eps:
                    # First timestamp in outputs, i.e., we need to extrapolate
                    # the input signal value.
                    timestamps.append(output_timestamps[j])
                    # Extrapolate input.
                    for var in input_var:
                        signals[var].append(signals[var][-1])
                    # Use output as is.
                    for var in output_var:
                        signals[var].append(output_signals[M[var]][j])
                    j += 1
                else:
                    # First timestamp in inputs, i.e., we need to extrapolate
                    # the output signal value.
                    timestamps.append(input_timestamps[i])
                    # Use input as is.
                    for var in input_var:
                        signals[var].append(input_signals[M[var]][i])
                    # Extrapolate output.
                    for var in output_var:
                        signals[var].append(signals[var][-1])
                    i += 1
        elif len(input_var) > 0:
            timestamps = input_timestamps
            signals = {var:input_signals[M[var]] for var in input_var}
        else:
            timestamps = output_timestamps
            signals = {var:output_signals[M[var]] for var in output_var}

        # This needs to be fetched at this point.
        horizon = self.spec.top.horizon

        if self.strict_horizon_check and horizon > timestamps[-1]:
            raise Exception("The horizon {} of the formula is too long compared to input signal length {}. The robustness cannot be computed.".format(horizon, timestamps[-1]))

        # Transform the STL formula to past temporal logic.
        try:
            self.spec.pastify()
        except:
            # TODO: Handle errors.
            raise

        # Scale the signals.
        for var in input_var:
            signals[var] = self.sut.scale_signal(signals[var], self.sut.input_range[M[var]], target_A=0, target_B=1) 
        for var in output_var:
            signals[var] = self.sut.scale_signal(signals[var], self.sut.output_range[M[var]], target_A=0, target_B=1) 

        # Build trajectories in appropriate form.
        trajectories = {var:signals[var] for var in formula_variables}
        trajectories["time"] = list(timestamps)

        robustness_signal = self.spec.evaluate(trajectories)

        robustness = None
        for t, r in robustness_signal:
            if t == horizon:
                robustness = r
                break

        if robustness is None:
            if self.strict_horizon_check:
                raise Exception("Could not figure out correct robustness at horizon {} from robustness signal {}.".format(horizon, robustness_signal))
            else:
                robustness = robustness_signal[-1][1]

        # Clip the robustness to [0, 1].
        if clip:
            robustness = max(0, min(robustness, 1))

        return robustness

    def __call__(self, r: SUTResult, *args, **kwargs):
        clip = kwargs["clip"] if "clip" in kwargs else True
        if r.output_timestamps is None:
            return self._evaluate_vector(r.outputs, clip=clip)
        else:
            return self._evaluate_signal(r, clip=clip)

