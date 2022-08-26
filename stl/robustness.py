import numpy as np

# TODO: Save computed robustness values for efficiency and implement reset for reuse.

class Traces:

    def __init__(self, timestamps, signals):
        self.timestamps = timestamps
        self.signals = signals

        # Check that all signals have correct length.
        for s in signals.values():
            if len(s) != len(timestamps):
                raise ValueError("All signals must have exactly as many samples as there are timestamps.")

    @classmethod
    def from_mixed_signals(C, *args, sampling_period=None):
        """Instantiate the class from signals that have different timestamps
        (with 0 as a first timestamp) and different lengths. This is done by
        finding the maximum signal length and using that as a signal length,
        dividing this length into pieces according to the sampling period
        (smallest observed difference between timestamps if None), and filling
        values by assuming constant value.

        The input is expected to be of the form
        name1, timestamps1, signal1, name2, timestamps2, signal2, ..."""

        """
        Currently the code does not work reliably with time steps lower than
        1e-4, but this should be enough.
        """

        if sampling_period is None:
            raise NotImplementedError()

        # Check that all timestamps begin with 0. Otherwise the code below is
        # not valid.
        for i in range(0, len(args), 3):
            if args[i+1][0] != 0:
                raise Exception("The first timestamp should be 0 in all signals.")

        # Maximum signal length.
        T = max(args[i+1][-1] for i in range(0, len(args), 3))

        # New timestamps.
        timestamps = [i*sampling_period for i in range(0, int(T/sampling_period) + 1)]

        # Check that all signals begin from time 0. Otherwise this does not work.

        # Fill the signals by assuming constant value.
        signals = {}
        eps = 1e-5
        for i in range(0, len(args), 3):
            name = args[i]
            signal_timestamps = args[i+1]
            signal_values = args[i+2]

            signals[name] = np.empty(shape=(len(timestamps)))
            pos = 0
            for n, t in enumerate(timestamps):
                while pos < len(signal_timestamps) and signal_timestamps[pos] <= t + eps:
                    pos += 1

                value = signal_values[pos - 1]
                signals[name][n] = value

        return C(timestamps, signals)

    def search_time_index(self, t, start=0):
        """Finds the index of the time t in the timestamps using binary
        search."""

        lower_idx = start
        upper_idx = len(self.timestamps) - 1
        middle = (lower_idx + upper_idx)//2
        while lower_idx <= upper_idx:
            if self.timestamps[middle] < t:
                lower_idx = middle + 1
            elif self.timestamps[middle] > t:
                upper_idx = middle - 1
            else:
                break

            middle = (lower_idx + upper_idx)//2

        if self.timestamps[middle] == t:
            return middle
        else:
            return -1

class TreeIterator:

    def __init__(self, node):
        self.nodes = [node]

    def __next__(self):
        try:
            node = self.nodes.pop(0)
            self.nodes += node.formulas

            return node
        except IndexError:
            raise StopIteration

class STL:
    """Base class for all logical operations and atoms."""

    def __iter__(self):
        return TreeIterator(self)

class Signal(STL):

    def __init__(self, name, range=None):
        self.formulas = []
        self.name = name
        self.range = range.copy() if range is not None else None
        self.horizon = 0

    def eval(self, traces, return_effective_range=True):
        if return_effective_range and self.range is not None:
            effective_range_signal = np.empty(shape=(len(traces.timestamps), 2))
            effective_range_signal[:] = np.array([self.range[0], self.range[1]]).reshape(1, 2)
        else:
            effective_range_signal = None
        # We return a copy so that subsequent robustness computations can
        # safely reuse arrays. We also enforce floats in order to avoid errors.
        return np.array(traces.signals[self.name], copy=True, dtype="float64"), effective_range_signal

class Constant(STL):

    def __init__(self, val):
        self.formulas = []
        self.val = val
        self.range = [val, val]
        self.horizon = 0

    def eval(self, traces, return_effective_range=True):
        if return_effective_range:
            effective_range_signal = np.full(shape=(len(traces.timestamps), 2), fill_value=self.val)
        else:
            effective_range_signal = None
        # We must always produce a new array because subsequent robustness
        # computations can reuse arrays.
        return np.full(len(traces.timestamps), self.val), effective_range_signal

class Sum(STL):

    def __init__(self, left_formula, right_formula):
        self.formulas = [left_formula, right_formula]

        if self.formulas[0].range is None or self.formulas[1].range is None:
            self.range = None
        else:
            A = self.formulas[0].range[0] + self.formulas[1].range[0]
            B = self.formulas[0].range[1] + self.formulas[1].range[1]
            self.range = [A, B]

        self.horizon = 0

    def eval(self, traces, return_effective_range=True):
        if return_effective_range and self.range is not None:
            effective_range_signal = np.empty(shape=(len(traces.timestamps), 2))
            effective_range_signal[:] = np.array([self.range[0], self.range[1]]).reshape(1, 2)
        else:
            effective_range_signal = None

        left_formula_robustness, _ = self.formulas[0].eval(traces, return_effective_range=False)
        right_formula_robustness, _ = self.formulas[1].eval(traces, return_effective_range=False)
        return np.add(left_formula_robustness, right_formula_robustness, out=left_formula_robustness), effective_range_signal

class Subtract(STL):

    def __init__(self, left_formula, right_formula):
        self.formulas = [left_formula, right_formula]

        if self.formulas[0].range is None or self.formulas[1].range is None:
            self.range = None
        else:
            A = self.formulas[0].range[0] - self.formulas[1].range[1]
            B = self.formulas[0].range[1] - self.formulas[1].range[0]
            self.range = [A, B]

        self.horizon = 0

    def eval(self, traces, return_effective_range=True):
        if return_effective_range and self.range is not None:
            effective_range_signal = np.empty(shape=(len(traces.timestamps), 2))
            effective_range_signal[:] = np.array([self.range[0], self.range[1]]).reshape(1, 2)
        else:
            effective_range_signal = None

        left_formula_robustness, _ = self.formulas[0].eval(traces, return_effective_range=False)
        right_formula_robustness, _ = self.formulas[1].eval(traces, return_effective_range=False)
        return np.subtract(left_formula_robustness, right_formula_robustness, out=left_formula_robustness), effective_range_signal

class Multiply(STL):

    def __init__(self, left_formula, right_formula):
        self.formulas = [left_formula, right_formula]
        if self.formulas[0].range is None or self.formulas[1].range is None:
            self.range = None
        else:
            A = self.formulas[0].range[0] * self.formulas[1].range[0]
            B = self.formulas[0].range[1] * self.formulas[1].range[1]
            self.range = [A, B]
        self.horizon = 0

    def eval(self, traces, return_effective_range=True):
        if return_effective_range and self.range is not None:
            effective_range_signal = np.empty(shape=(len(traces.timestamps), 2))
            effective_range_signal[:] = np.array([self.range[0], self.range[1]]).reshape(1, 2)
        else:
            effective_range_signal = None

        left_formula_robustness, _ = self.formulas[0].eval(traces, return_effective_range=False)
        right_formula_robustness, _ = self.formulas[1].eval(traces, return_effective_range=False)
        return np.multiply(left_formula_robustness, right_formula_robustness, out=left_formula_robustness), effective_range_signal

class Divide(STL):

    def __init__(self, left_formula, right_formula):
        self.formulas = [left_formula, right_formula]
        if self.formulas[0].range is None or self.formulas[1].range is None:
            self.range = None
        else:
            if self.formulas[1].range[1] == 0:
                raise Exception("Cannot determine a finite range for division as the right formula upper bound is 0.")
            if self.formulas[1].range[0] == 0:
                raise Exception("Cannot determine a finite range for division as the right formula lower bound is 0.")
            A = self.formulas[0].range[0] / self.formulas[1].range[1]
            B = self.formulas[0].range[1] / self.formulas[1].range[0]
            self.range = [A, B]
        self.horizon = 0

    def eval(self, traces, return_effective_range=True):
        if return_effective_range and self.range is not None:
            effective_range_signal = np.empty(shape=(len(traces.timestamps), 2))
            effective_range_signal[:] = np.array([self.range[0], self.range[1]]).reshape(1, 2)
        else:
            effective_range_signal = None

        left_formula_robustness, _ = self.formulas[0].eval(traces, return_effective_range=False)
        right_formula_robustness, _ = self.formulas[1].eval(traces, return_effective_range=False)
        return np.divide(left_formula_robustness, right_formula_robustness, out=left_formula_robustness), effective_range_signal

class GreaterThan(STL):

    def __init__(self, left_formula, right_formula):
        if isinstance(left_formula, (int, float)):
            left_formula = Constant(left_formula)
        if isinstance(right_formula, (int, float)):
            right_formula = Constant(right_formula)
        self.formulas = [left_formula, right_formula]

        if self.formulas[0].range is None or self.formulas[1].range is None:
            self.range = None
        else:
            A = self.formulas[0].range[0] - self.formulas[1].range[1]
            B = self.formulas[0].range[1] - self.formulas[1].range[0]
            self.range = [A, B]

        self.horizon = 0

    def eval(self, traces, return_effective_range=True):
        if return_effective_range and self.range is not None:
            effective_range_signal = np.empty(shape=(len(traces.timestamps), 2))
            effective_range_signal[:] = np.array([self.range[0], self.range[1]]).reshape(1, 2)
        else:
            effective_range_signal = None

        left_formula_robustness, _ = self.formulas[0].eval(traces, return_effective_range=False)
        right_formula_robustness, _ = self.formulas[1].eval(traces, return_effective_range=False)
        return np.subtract(left_formula_robustness, right_formula_robustness, out=left_formula_robustness), effective_range_signal

class LessThan(STL):

    def __init__(self, left_formula, right_formula):
        if isinstance(left_formula, (int, float)):
            left_formula = Constant(left_formula)
        if isinstance(right_formula, (int, float)):
            right_formula = Constant(right_formula)
        self.formulas = [left_formula, right_formula]

        if self.formulas[0].range is None or self.formulas[1].range is None:
            self.range = None
        else:
            A = self.formulas[1].range[0] - self.formulas[0].range[1]
            B = self.formulas[1].range[1] - self.formulas[0].range[0]
            self.range = [A, B]

        self.horizon = 0

    def eval(self, traces, return_effective_range=True):
        if return_effective_range and self.range is not None:
            effective_range_signal = np.empty(shape=(len(traces.timestamps), 2))
            effective_range_signal[:] = np.array([self.range[0], self.range[1]]).reshape(1, 2)
        else:
            effective_range_signal = None

        left_formula_robustness, _ = self.formulas[0].eval(traces, return_effective_range=False)
        right_formula_robustness, _ = self.formulas[1].eval(traces, return_effective_range=False)
        return np.subtract(right_formula_robustness, left_formula_robustness, out=right_formula_robustness), effective_range_signal

class Abs(STL):

    def __init__(self, formula):
        self.formulas = [formula]
        if self.formulas[0].range is None:
            self.range = None
        else:
            A = self.formulas[0].range[0]
            B = self.formulas[0].range[1]
            if A <= 0:
              if B > 0:
                self.range = [0, B]
              else:
                self.range = [-B, -A]
            else:
              self.range = [A, B]

        self.horizon = 0

    def eval(self, traces, return_effective_range=True):
        if return_effective_range and self.range is not None:
            effective_range_signal = np.empty(shape=(len(traces.timestamps), 2))
            effective_range_signal[:] = np.array([self.range[0], self.range[1]]).reshape(1, 2)
        else:
            effective_range_signal = None

        formula_robustness, _ = self.formulas[0].eval(traces, return_effective_range=False)
        return np.abs(formula_robustness, out=formula_robustness), effective_range_signal

class Equals(STL):

    def __init__(self, left_formula, right_formula):
        self.formulas = [left_formula, right_formula]
        self.formula_robustness = Not(Abs(Subtract(self.formulas[0], self.formulas[1])))

        if self.formulas[0].range is None or self.formulas[1].range is None:
            self.range = None
        else:
            A = self.formulas[0].range[0] - self.formulas[1].range[1]
            B = self.formulas[0].range[1] - self.formulas[1].range[0]
            if A >= 0:
                self.range = [-B, -A]
            elif A < 0 and B >= 0:
                self.range = [-max(-A, B), 0]
            else:
                self.range = [A, B]
            # Make sure that 1 is included in the interval.
            if self.range[0] > 1:
                self.range[0] = 1
            elif self.range[1] < 1:
                self.range[1] = 1

        self.horizon = 0

    def eval(self, traces, return_effective_range=True):
        if return_effective_range and self.range is not None:
            effective_range_signal = np.empty(shape=(len(traces.timestamps), 2))
            effective_range_signal[:] = np.array([self.range[0], self.range[1]]).reshape(1, 2)
        else:
            effective_range_signal = None

        robustness, _ = self.formula_robustness.eval(traces, return_effective_range=False)
        return np.where(robustness == 0, 1, robustness), effective_range_signal

class Next(STL):

    def __init__(self, formula):
        self.formulas = [formula]
        self.range = self.formulas[0].range.copy() if self.formulas[0].range is not None else None
        self.horizon = 1 + self.formulas[0].horizon

    def eval(self, traces, return_effective_range=True):
        formula_robustness, formula_effective_range_signal = self.formulas[0].eval(traces, return_effective_range)
        robustness = np.roll(formula_robustness, -1)[:-1]
        if return_effective_range and self.range is not None:
            effective_range_signal = np.roll(formula_effective_range_signal, -1)[:-1]
        else:
            effective_range_signal = None

        return robustness, formula_effective_range

class Global(STL):

    def __init__(self, lower_time_bound, upper_time_bound, formula):
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.formulas = [formula]
        self.range = None if self.formulas[0].range is None else self.formulas[0].range.copy()
        self.horizon = self.upper_time_bound + self.formulas[0].horizon

    def eval(self, traces, return_effective_range=True):
        formula_robustness, formula_effective_range_signal = self.formulas[0].eval(traces, return_effective_range)
        robustness = np.empty(shape=(len(formula_robustness)))
        if return_effective_range and formula_effective_range_signal is not None:
            effective_range_signal = np.empty(shape=(len(formula_effective_range_signal), 2))
        # We save the found positions as most often we use integer timestamps and
        # evenly sampled signals, so this has huge speed benefit.
        prev_lower_bound_pos = len(traces.timestamps) - 1
        prev_upper_bound_pos = len(traces.timestamps) - 1
        prev_min = float("inf")
        prev_min_idx = len(traces.timestamps)
        for current_time_pos in range(len(traces.timestamps) - 1, -1, -1):
            # Lower and upper times for the current time.
            lower_bound = traces.timestamps[current_time_pos] + self.lower_time_bound
            upper_bound = traces.timestamps[current_time_pos] + self.upper_time_bound

            # Find the corresponding positions in timestamps.
            if lower_bound > traces.timestamps[-1]:
                lower_bound_pos = len(traces.timestamps) - 1
            else:
                if traces.timestamps[prev_lower_bound_pos - 1] == lower_bound:
                    lower_bound_pos = prev_lower_bound_pos - 1
                else:
                    lower_bound_pos = traces.search_time_index(lower_bound, start=current_time_pos)
                    # TODO: This should never happen except for floating point
                    # inaccuracies. We now raise an exception as otherwise the
                    # user gets unexpected behavior.
                    if lower_bound_pos < 0:
                        raise Exception("No timestamp '{}' found even though it should exist.".format(lower_bound))

            if upper_bound > traces.timestamps[-1]:
                upper_bound_pos = len(traces.timestamps) - 1
            else:
                if traces.timestamps[prev_upper_bound_pos - 1] == upper_bound:
                    upper_bound_pos = prev_upper_bound_pos - 1
                else:
                    upper_bound_pos = traces.search_time_index(upper_bound, start=lower_bound_pos)
                    # See above.
                    if upper_bound_pos < 0:
                        raise Exception("No timestamp '{}' found even though it should exist.".format(upper_bound))

            # Find minimum between the positions.
            start_pos = lower_bound_pos
            end_pos = upper_bound_pos + 1
            # Check if the previous minimum was found in the overlap
            # between the current search area and the previous one. If so,
            # then adjust the upper bound by removing the overlap from the
            # end.
            if prev_min_idx < end_pos:
                end_pos -= end_pos - prev_lower_bound_pos

            if start_pos < end_pos:
                min_idx = start_pos + np.argmin(formula_robustness[start_pos: end_pos])
                if prev_min_idx > upper_bound_pos or formula_robustness[min_idx] < prev_min:
                    prev_min_idx = min_idx
                    prev_min = formula_robustness[min_idx]

            prev_lower_bound_pos = start_pos
            prev_upper_bound_pos = end_pos - 1

            robustness[current_time_pos] = prev_min
            if return_effective_range and formula_effective_range_signal is not None:
                effective_range_signal[current_time_pos] = formula_effective_range_signal[prev_min_idx]

        return robustness, effective_range_signal if return_effective_range and formula_effective_range_signal is not None else None

class Finally(STL):

    def __init__(self, lower_time_bound, upper_time_bound, formula):
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.formulas = [formula]
        self.formula_robustness = Not(Global(self.lower_time_bound, self.upper_time_bound, Not(self.formulas[0])))
        self.range = self.formula_robustness.range.copy() if self.formula_robustness.range is not None else None
        self.horizon = self.upper_time_bound + self.formulas[0].horizon

    def eval(self, traces, return_effective_range=True):
        return self.formula_robustness.eval(traces, return_effective_range)

class Not(STL):

    def __init__(self, formula):
        self.formulas = [formula]
        if self.formulas[0].range is None:
            self.range = None
        else:
            self.range = [-1*self.formulas[0].range[1], -1*self.formulas[0].range[0]]

        self.horizon = self.formulas[0].horizon

    def eval(self, traces, return_effective_range=True):
        formula_robustness, formula_effective_range_signal = self.formulas[0].eval(traces, return_effective_range)
        if return_effective_range and formula_effective_range_signal is not None:
            np.multiply(-1, formula_effective_range_signal, out=formula_effective_range_signal)
            effective_range_signal = np.roll(formula_effective_range_signal, -1, axis=0)
        else:
            effective_range_signal = None

        return np.multiply(-1, formula_robustness, out=formula_robustness), effective_range_signal

class Implication(STL):

    def __init__(self, left_formula, right_formula):
        self.formulas = [left_formula, right_formula]

        if self.formulas[0].range is None or self.formulas[1].range is None:
            self.range = None
        else:
            A = max(-self.formulas[0].range[1], self.formulas[1].range[0])
            B = max(-self.formulas[0].range[0], self.formulas[1].range[1])
            self.range = [A, B]

        self.horizon = max(self.formulas[0].horizon, self.formulas[1].horizon)

    def eval(self, traces, return_effective_range=True):
        return Or(Not(self.formulas[0]), self.formulas[1]).eval(traces, return_effective_range)

class Or(STL):

    def __init__(self, *args, nu=None):
        self.formulas = list(args)
        # We save the actual definition in another attribute in order to work
        # correctly with nonassociativity and the parser.
        self.formula_robustness = Not(And(*[Not(f) for f in self.formulas]))
        self.range = self.formula_robustness.range.copy() if self.formula_robustness.range is not None else None
        self.horizon = self.formula_robustness.horizon

    def eval(self, traces, return_effective_range=True):
        return self.formula_robustness.eval(traces, return_effective_range)

class And(STL):

    def __init__(self, *args, nu=None):
        self.formulas = list(args)
        self.nu = nu
        
        if self.nu is not None and self.nu <= 0:
            raise ValueError("The nu parameter must be positive.")

        self.horizon = max(f.horizon for f in self.formulas)

        A = [f.range[0] if f.range is not None else None for f in self.formulas]
        B = [f.range[1] if f.range is not None else None for f in self.formulas]
        if None in A or None in B:
            self.range = None
        else:
            self.range = [min(A), min(B)]

    def eval(self, traces, return_effective_range=True):
        if self.nu is None:
            return self._eval_traditional(traces, return_effective_range)
        else:
            return self._eval_alternative(traces, self.nu, return_effective_range)

    def _eval_traditional(self, traces, return_effective_range):
        """This is the usual and."""

        # Evaluate the robustness of all subformulas and save the robustness
        # signals into one 2D array.
        M = len(self.formulas)
        ranges_initialized = False
        for i in range(M):
            formula_robustness, formula_range_signal = self.formulas[i].eval(traces, return_effective_range)
            if i == 0:
                rho = np.empty(shape=(M, len(formula_robustness)))
                if formula_range_signal is not None:
                    bounds = np.empty(shape=(M, formula_range_signal.shape[0], 2))
                    ranges_initialized = True

            rho[i,:] = formula_robustness
            if ranges_initialized:
                if formula_range_signal is not None:
                    bounds[i,:] = formula_range_signal
                else:
                    del bounds
                    ranges_initialized = False

        if ranges_initialized:
            min_idx = np.argmin(rho, axis=0)
            return rho[min_idx,np.arange(len(min_idx))], bounds[min_idx,np.arange(len(min_idx))]
        else:
            return np.min(rho, axis=0), None

    def _eval_alternative(self, traces, nu, return_effective_range=True):
        """This is the alternative and."""

        # Evaluate the robustness of all subformulas and save the robustness
        # signals into one 2D array.
        M = len(self.formulas)
        ranges_initialized = False
        for i in range(M):
            formula_robustness, formula_range_signal = self.formulas[i].eval(traces, return_effective_range)
            if i == 0:
                rho = np.empty(shape=(M, len(formula_robustness)))
                if return_effective_range and formula_range_signal is not None:
                    bounds = np.empty(shape=(M, formula_range_signal.shape[0], 2))
                    ranges_initialized = True

            rho[i,:] = formula_robustness
            if ranges_initialized:
                if formula_range_signal is not None:
                    bounds[i,:] = formula_range_signal
                else:
                    del bounds
                    ranges_initialized = False

        rho_argmin = np.argmin(rho, axis=0)

        robustness = np.empty(shape=(len(rho_argmin)))
        if ranges_initialized:
            range_signal = np.empty(shape=(len(rho_argmin), 2))
        for i in range(len(rho_argmin)):
            j = rho_argmin[i]

            if rho[j,i] == 0:
                robustness[i] = 0

                if ranges_initialized:
                    range_signal[i,0] = 0
                    range_signal[i,1] = 0
            else:
                # TODO: Would it make sense to compute rho_tilde for the complete
                # x-axis in advance? Does it use potentially much memory for no
                # speedup?
                rho_tilde = rho[:,i]/rho[j,i] - 1
                if rho[j,i] < 0:
                    weights = np.exp(np.multiply(nu, rho_tilde))
                    weighted = np.multiply(rho[j,i], np.exp(rho_tilde))
                elif rho[j,i] > 0:
                    weights = np.exp(np.multiply(-nu, rho_tilde))
                    weighted = rho[:,i]

                """
                if i in [0, 10, 20, 30, 40]:
                    print("weights: {}".format(weights/np.sum(weights)))
                """
                robustness[i] = np.dot(weights, weighted) / np.sum(weights)

                if ranges_initialized:
                    range_signal[i,0] = np.dot(weights, bounds[:,i,0]) / np.sum(weights)
                    range_signal[i,1] = np.dot(weights, bounds[:,i,1]) / np.sum(weights)

        return robustness, range_signal if ranges_initialized else None

# TODO: Implement these.
StrictlyLessThan = LessThan
StrictlyGreaterThan = GreaterThan

