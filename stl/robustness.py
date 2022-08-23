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

    def eval(self, traces, time=None):
        # We return a copy so that subsequent robustness computations can
        # safely reuse arrays. We also enforce floats in order to avoid errors.
        effective_range = self.range if time is not None else None
        return np.array(traces.signals[self.name], copy=True, dtype="float64"), effective_range

class Constant(STL):

    def __init__(self, val):
        self.formulas = []
        self.val = val
        self.range = [val, val]
        self.horizon = 0

    def eval(self, traces, time=None):
        # We must always produce a new array because subsequent robustness
        # computations can reuse arrays.
        effective_range = self.range if time is not None else None
        return np.full(len(traces.timestamps), self.val), effective_range

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

    def eval(self, traces, time=None):
        left_formula_robustness, _ = self.formulas[0].eval(traces, time)
        right_formula_robustness, _ = self.formulas[1].eval(traces, time)
        effective_range = self.range if time is not None else None
        return np.add(left_formula_robustness, right_formula_robustness, out=left_formula_robustness), effective_range

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

    def eval(self, traces, time=None):
        left_formula_robustness, _ = self.formulas[0].eval(traces, time)
        right_formula_robustness, _ = self.formulas[1].eval(traces, time)
        effective_range = self.range if time is not None else None
        return np.subtract(left_formula_robustness, right_formula_robustness, out=left_formula_robustness), effective_range

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

    def eval(self, traces, time=None):
        left_formula_robustness, _ = self.formulas[0].eval(traces, time)
        right_formula_robustness, _ = self.formulas[1].eval(traces, time)
        effective_range = self.range if time is not None else None
        return np.multiply(left_formula_robustness, right_formula_robustness, out=left_formula_robustness), effective_range

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

    def eval(self, traces, time=None):
        left_formula_robustness, _ = self.formulas[0].eval(traces, time)
        right_formula_robustness, _ = self.formulas[1].eval(traces, time)
        effective_range = self.range if time is not None else None
        return np.subtract(left_formula_robustness, right_formula_robustness, out=left_formula_robustness), effective_range

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

    def eval(self, traces, time=None):
        left_formula_robustness, _ = self.formulas[0].eval(traces, time)
        right_formula_robustness, _ = self.formulas[1].eval(traces, time)
        effective_range = self.range if time is not None else None
        return np.subtract(right_formula_robustness, left_formula_robustness, out=right_formula_robustness), effective_range

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

    def eval(self, traces, time=None):
        formula_robustness, _ = self.formulas[0].eval(traces, time)
        effective_range = self.range if time is not None else None
        return np.abs(formula_robustness, out=formula_robustness), effective_range

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

    def eval(self, traces, time=None):
        robustness, _ = self.formula_robustness.eval(traces)
        effective_range = self.range if time is not None else None
        return np.where(robustness == 0, 1, robustness), effective_range

class Next(STL):

    def __init__(self, formula):
        self.formulas = [formula]
        self.range = self.formulas[0].range.copy() if self.formulas[0].range is not None else None
        self.horizon = 1 + self.formulas[0].horizon

    def eval(self, traces, time=None):
        formula_robustness, formula_effective_range = self.formulas[0].eval(traces, time)
        res = np.roll(formula_robustness, -1)
        return res[:-1], formula_effective_range

class Global(STL):

    def __init__(self, lower_time_bound, upper_time_bound, formula):
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.formulas = [formula]
        self.range = None if self.formulas[0].range is None else self.formulas[0].range.copy()
        self.horizon = self.upper_time_bound + self.formulas[0].horizon

    def eval(self, traces, time=None):
        robustness, formula_effective_range = self.formulas[0].eval(traces, time)
        result = np.empty(shape=(len(robustness)))
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
                min_idx = start_pos + np.argmin(robustness[start_pos: end_pos])
                if prev_min_idx > upper_bound_pos or robustness[min_idx] < prev_min:
                    prev_min_idx = min_idx
                    prev_min = robustness[min_idx]

            prev_lower_bound_pos = start_pos
            prev_upper_bound_pos = end_pos - 1

            result[current_time_pos] = prev_min

        return result, formula_effective_range

class Finally(STL):

    def __init__(self, lower_time_bound, upper_time_bound, formula):
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.formulas = [formula]
        self.formula_robustness = Not(Global(self.lower_time_bound, self.upper_time_bound, Not(self.formulas[0])))
        self.range = self.formula_robustness.range.copy() if self.formula_robustness.range is not None else None
        self.horizon = self.upper_time_bound + self.formulas[0].horizon

    def eval(self, traces, time=None):
        return self.formula_robustness.eval(traces, time)

class Not(STL):

    def __init__(self, formula):
        self.formulas = [formula]
        if self.formulas[0].range is None:
            self.range = None
        else:
            self.range = [-1*self.formulas[0].range[1], -1*self.formulas[0].range[0]]

        self.horizon = self.formulas[0].horizon

    def eval(self, traces, time=None):
        formula_robustness, formula_effective_range = self.formulas[0].eval(traces, time)
        if formula_effective_range is not None:
            effective_range = [-formula_effective_range[1], -formula_effective_range[0]]
        else:
            effective_range = None
        return np.multiply(-1, formula_robustness, out=formula_robustness), effective_range

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

    def eval(self, traces, time=None):
        return Or(Not(self.formulas[0]), self.formulas[1]).eval(traces, time)

class Or(STL):

    def __init__(self, *args, nu=None):
        self.formulas = list(args)
        # We save the actual definition in another attribute in order to work
        # correctly with nonassociativity and the parser.
        self.formula_robustness = Not(And(*[Not(f) for f in self.formulas]))
        self.range = self.formula_robustness.range.copy() if self.formula_robustness.range is not None else None
        self.horizon = self.formula_robustness.horizon

    def eval(self, traces, time=None):
        return self.formula_robustness.eval(traces, time)

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

    def eval(self, traces, time=None):
        if self.nu is None:
            return self._eval_traditional(traces, time)
        else:
            return self._eval_alternative(traces, self.nu, time)

    def _eval_traditional(self, traces, time=None):
        """This is the usual and."""

        # Evaluate the robustness of all subformulas and save the robustness
        # signals into one 2D array.
        M = len(self.formulas)
        for i in range(M):
            r, effective_range = self.formulas[i].eval(traces, time)
            if i == 0:
                rho = np.empty(shape=(M, len(r)))
                effective_ranges = []
            rho[i,:] = r
            effective_ranges.append(effective_range)

        if time is not None:
            idx = np.argmin(rho[:,time])
            effective_range = effective_ranges[idx]
        else:
            effective_range = None
        return np.min(rho, axis=0), effective_range

    def _eval_alternative(self, traces, nu, time=None):
        """This is the alternative and."""

        # Evaluate the robustness of all subformulas and save the robustness
        # signals into one 2D array.
        M = len(self.formulas)
        for i in range(M):
            r = self.formulas[i].eval(traces)
            if i == 0:
                rho = np.empty(shape=(M, len(r)))
            rho[i,:] = r

        rho_min = np.min(rho, axis=0)

        robustness = np.empty(shape=(len(rho_min)))
        for i in range(len(rho_min)):
            # TODO: Would it make sense to compute rho_tilde for the complete
            # x-axis in advance? Does it use potentially much memory for no
            # speedup?
            rho_tilde = rho[:,i]/rho_min[i] - 1

            if rho_min[i] < 0:
                numerator = rho_min[i] * np.sum(np.exp((1+nu) * rho_tilde))
                denominator = np.sum(np.exp(nu * rho_tilde))
            elif rho_min[i] > 0:
                numerator = np.sum(np.multiply(rho[:,i], np.exp(-nu * rho_tilde)))
                denominator = np.sum(np.exp(-1 * nu * rho_tilde))
            else:
                numerator = 0
                denominator = 1

            robustness[i] = numerator/denominator
        
        return robustness

# TODO: Implement these.
StrictlyLessThan = LessThan
StrictlyGreaterThan = GreaterThan

