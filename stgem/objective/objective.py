#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
All objectives must output a single number. If several outputs are required
for a single input, multiple objectives must be specified.
"""

import numpy as np

class Objective:

    def __init__(self):
        self.parameters = {}

    def setup(self, sut):
        self.sut = sut

    def __getattr__(self, name):
        if "parameters" in self.__dict__:
            if name in self.parameters:
                return self.parameters.get(name)

        raise AttributeError(name)

    def __call__(self, t, r):
        raise NotImplementedError

class Minimize(Objective):
    """Objective function which selects the minimum of the specified components
    for vector outputs and minimum value of the selected signals for signal
    outputs."""

    def __init__(self, selected=None, scale=False, invert=False, clip=True):
        super().__init__()
        if not (isinstance(selected, list) or isinstance(selected, tuple) or selected is None):
            raise Exception("The parameter 'selected' must be None or a list or a tuple.")

        self.parameters["selected"] = selected
        self.parameters["scale"] = scale
        self.parameters["invert"] = invert
        self.parameters["clip"] = clip

    def __call__(self, t, r):
        idx = self.selected if self.selected is not None else list(range(len(r.outputs)))
        if r.output_timestamps is not None:
            # Find the minimum value of the selected signals and form a vector
            # from them.
            v = np.array([min(signal) for signal in r.outputs[idx,:]])
        else:
            # Just select the desired comporents.
            v = r.outputs[idx]

        if self.invert:
            v = v*(-1)
            ranges = np.asarray([[-self.sut.output_range[i][1], -self.sut.output_range[i][0]] for i in idx])
        else:
            ranges = [self.sut.output_range[i] for i in idx]

        if self.scale:
            output = self.sut.scale(v.reshape(1, -1), ranges, target_A=0, target_B=1).reshape(-1)
        else:
            output = v

        if self.clip:
            return max(0, min(1, min(output)))
        else:
            return min(output)

class FalsifySTL(Objective):
    """Objective function to falsify a STL specification. By default the
    robustness is not scaled, but if scale is True and variable ranges have
    been specified for the signals, then the robustness is scaled to
    [0, 1].

    The parameter strict_horizon_check controls if an exception is raised if
    the signal is too short to determine the truth value of the specification.
    If False and the signal is too short, then the best estimate for the
    robustness is returned, but this value might be incorrect if the signal is
    augmented with appropriate values.

    The parameter epsilon is a value which is added to positive robustness
    values. A positive epsilon value thus makes falsification harder. This is
    sometimes useful if the observed values are very close to 0 but positive
    and the machine learning models consider such a value to be 0. Raising the
    bar a bit can encourage the models to work harder and eventually produce
    robustness which is nonpositive."""

    def __init__(self, specification, epsilon=0, scale=False, strict_horizon_check=True):
        super().__init__()

        self.dim = 1
        self.specification = specification
        self.parameters["epsilon"] = epsilon
        self.parameters["scale"] = scale
        if self.scale and self.specification.var_range is None:
            raise Exception("The specification does not include a range for robustness. This is needed for scaling.")
        self.parameters["strict_horizon_check"] = strict_horizon_check

    def setup(self, sut):
        super().setup(sut)

        try:
            import tltk_mtl as STL
        except:
            raise

        if not isinstance(self.specification, STL.TLTK_MTL):
            raise Exception("Expected specification to be TLTK class not '{}'".format(type(self.specification)))

        self.horizon = self.specification.horizon
        self.formula_variables = self.specification.variables

        # Find the objects with time bounds in the formula.
        # TODO: An iterator would be nice in TLTk for this.
        def bounded(formula):
            if isinstance(formula, (STL.Predicate, STL.Signal)) or formula is None:
                return []
            elif isinstance(formula, (STL.Global, STL.Finally)):
                return [formula] + bounded(formula.subformula)
            elif isinstance(formula, STL.Until):
                return [formula] + bounded(formula.left_subformula) + bounded(formula.right_subformula)
            elif not isinstance(formula, STL.TLTK_MTL):
                raise Exception("Expected TLTK class not '{}' in time bounded object lookup.".format(type(formula)))
            elif formula.arity == 1:
                return bounded(formula.subformula)
            else: #formula.arity == 2:
                return bounded(formula.left_subformula) + bounded(formula.right_subformula)

        # Sampling period equals the minimum of the smallest positive time
        # bound referred to in the formula divided by K and the sampling step
        # of the SUT (if it exists). Compute the first number.
        K = 10
        self.time_bounded = bounded(self.specification)
        first = 1
        for x in self.time_bounded:
            if x.lower_time_bound > 0 and x.lower_time_bound < first:
                first = x.lower_time_bound
            if x.upper_time_bound > 0 and x.upper_time_bound < first:
                first = x.upper_time_bound
        first /= K
        if hasattr(self.sut, "sampling_step"):
            self.sampling_period = min(first, self.sut.sampling_step)
        else:
            self.sampling_period = first
        from math import log10, floor
        self.precision = abs(floor(log10(self.sampling_period)))

        # Create a mapping for an easy access to correct signal. Save which
        # variables refer to input signals and which to output signals.
        self.M = {}
        self.input_variables = []
        self.output_variables = []
        for var in self.formula_variables:
            try:
                idx = self.sut.outputs.index(var)
                self.output_variables.append(var)
                self.M[var] = ["output", idx]
            except ValueError:
                try:
                    idx = self.sut.inputs.index(var)
                    self.input_variables.append(var)
                    self.M[var] = ["input", idx]
                except ValueError:
                    raise Exception("Variable '{}' not in input or output variables.".format(var))

    def adjust_time_bounds(self):
        for x in self.time_bounded:
            x.old_lower_time_bound = x.lower_time_bound
            x.old_upper_time_bound = x.upper_time_bound
            x.lower_time_bound = int(x.lower_time_bound / self.sampling_period)
            x.upper_time_bound = int(x.upper_time_bound / self.sampling_period)

    def reset_time_bounds(self):
        for x in self.time_bounded:
            x.lower_time_bound = x.old_lower_time_bound
            x.upper_time_bound = x.old_upper_time_bound

    def _evaluate_vector(self, test, output):
        # We assume that the output is a single observation of a signal. It
        # follows that not all STL formulas have a clear interpretation (like
        # always[0,30](x1 > 0 and x2 > 0). It is up to the user to ensure a
        # reasonable interpretation.

        self.specification.reset()

        timestamps = np.array([0], dtype=np.float32)
        trajectories = {}
        for var in self.formula_variables:
            if self.M[var][0] == "output":
                trajectories[var] = np.array([output[self.M[var][1]]], dtype=np.float32)
            else:
                trajectories[var] = np.array([test[self.M[var][1]]], dtype=np.float32)

        # Notice that the return value is a Cython MemoryView.
        robustness_signal = self.specification.eval_interval(trajectories, timestamps)

        robustness = robustness_signal[0]

        # Scale the robustness to [0, 1] if required.
        if self.scale:
            if robustness < 0:
                robustness = 0
            else:
                B = self.specification.var_range[1]
                robustness *= 1/B
                robustness += self.epsilon
                robustness = min(1, robustness)

        return robustness

    def _evaluate_signal(self, test, result):
        input_timestamps = test.input_timestamps
        output_timestamps = result.output_timestamps
        input_signals = test.input_denormalized
        output_signals = result.outputs

        """
        Here we find the robustness at time 0.

        We assume that the user guarantees that time is increasing. Floating
        point numbers and the current TLTK do not go well together. The reason
        is that intervall TLTK calls a function search_sorted to look for
        timestamps in an array. The timestamp must be an equal match (0.1
        != 0.1000000001) or otherwise it silently fails and returns the last
        timestamp. This can lead to errorneous results. The solution is to use
        integer timestamps.

        Another problem is that the differences between timestamps (input or
        output) can be very small and variable. We cannot simply take the
        minimum difference and use this as a sampling step because this can
        lead into very small time steps and thus to very long augmented
        signals. Very small time steps could be unnecessary too as the STL
        formula might only refer to relatively big time steps.

        Our solution to the above is to figure out the smallest timestep
        referred to in the STL formula (this is already figured out in the
        setup method) and divide this K equal pieces (currently K = 10). The
        minimum timestep equals one time unit. This determines a sampling
        period and we sample all signals according to this sampling period.
        This can mean discarding signal values or augmenting a signal.
        Currently we augment a signal by assuming a constant value.

        Using integer timestamps then requires scaling the time intervals
        occurring in the specification formulas. This as already done in the
        setup method.

        Currently the code does not work reliably with time steps lower than
        1e-4, but this should be enough.
        """

        # This check is necessary for validity.
        if output_timestamps[0] != 0 or (input_timestamps is not None and input_timestamps[0] != 0):
            raise Exception("The first timestamp should be 0 in both input and output signals.")

        T = max(output_timestamps[-1], 0 if input_timestamps is None else input_timestamps[-1])
        # Round to the same scale as the sampling period.
        T = round(T, self.precision)
        timestamps = [i*self.sampling_period for i in range(0, int(T/self.sampling_period) + 1)]

        # Fill in missing signal values for new timestamps by assuming constant
        # value.
        # TODO: Don't do extra work if it can be avoided.
        eps = 1e-5
        signals = {var:[] for var in self.formula_variables}
        for var_set, signal_timestamps in [(self.output_variables, output_timestamps)] + ([] if input_timestamps is None else [(self.input_variables, input_timestamps)]):
            for var in var_set:
                pos = 0
                for t in timestamps:
                    while pos < len(signal_timestamps) and signal_timestamps[pos] <= t + eps:
                        pos += 1

                    if self.M[var][0] == "output":
                        value = output_signals[self.M[var][1]][pos - 1]
                    else:
                        value = input_signals[self.M[var][1]][pos - 1]
                    signals[var].append(value)

                    if pos > len(signal_timestamps):
                        break

        self.specification.reset()

        # Allow slight inaccuracy in horizon check.
        if self.strict_horizon_check and self.horizon - 1e-2 > timestamps[-1]:
            raise Exception("The horizon {} of the formula is too long compared to signal length {}. The robustness cannot be computed.".format(self.horizon, timestamps[-1]))

        # Adjust time bounds.
        self.adjust_time_bounds()

        # Build trajectories in appropriate form.
        # Notice that converting to float32 (as required by TLTK) might lose
        # some precision. This should be of no consequence regarding
        # falsification.
        trajectories = {var:np.asarray(signals[var], dtype=np.float32) for var in self.formula_variables}

        # Use integer timestamps.
        timestamps = np.array(list(range(len(timestamps))), dtype=np.float32)
        # Notice that the return value is a Cython MemoryView.
        robustness_signal = self.specification.eval_interval(trajectories, timestamps)

        robustness = robustness_signal[0]

        # Reset time bounds. This allows reusing the specifications.
        self.reset_time_bounds()

        # Scale the robustness to [0, 1] if required.
        if self.scale:
            if robustness < 0:
                robustness = 0
            else:
                B = self.specification.var_range[1]
                robustness *= 1/B
                robustness += self.epsilon
                robustness = min(1, robustness)

        return robustness

    def __call__(self, t, r):
        if r.output_timestamps is None:
            return self._evaluate_vector(t.inputs, r.outputs)
        else:
            return self._evaluate_signal(t, r)

    @staticmethod
    def GreaterThan(A, B, C, D, left_signal=None, right_signal=None):
        """Static helper method for stating A*L + B >= C*R + D where L and R
        respectively are the left and right signals."""

        try:
            import tltk_mtl as STL
        except:
            raise

        return STL.LessThan(C, D, A, B, right_signal, left_signal)

    @staticmethod
    def StrictlyLessThan(A, B, C, D, left_signal=None, right_signal=None):
        """Static helper method for stating A*L + B < C*R + D where L and R
        respectively are the left and right signals."""

        try:
            import tltk_mtl as STL
        except:
            raise

        return STL.Not(FalsifySTL.GreaterThan(A, B, C, D, left_signal, right_signal))

    @staticmethod
    def StrictlyGreaterThan(A, B, C, D, left_signal=None, right_signal=None):
        """Static helper method for stating A*L + B > C*R + D where L and R
        respectively are the left and right signals."""

        return FalsifySTL.StrictlyLessThan(C, D, A, B, right_signal, left_signal)

