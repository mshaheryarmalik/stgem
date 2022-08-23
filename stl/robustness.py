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

    def eval(self, traces):
        return traces.signals[self.name]

class Constant(STL):

    def __init__(self, val):
        self.formulas = []
        self.val = val
        self.range = [val, val]
        self.horizon = 0

    def eval(self, traces):
        return np.full(len(traces.timestamps), self.val)

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

    def eval(self, traces):
        res = []
        left_formula_robustness = self.formulas[0].eval(traces)
        right_formula_robustness = self.formulas[1].eval(traces)
        return np.add(left_formula_robustness, right_formula_robustness)

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

    def eval(self, traces):
        return np.subtract(self.formulas[0].eval(traces), self.formulas[1].eval(traces))

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

    def eval(self, traces):
        return Subtract(self.formulas[0], self.formulas[1]).eval(traces)

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

    def eval(self, traces):
        res = []
        left_formula_robustness = self.formulas[0].eval(traces)
        right_formula_robustness = self.formulas[1].eval(traces)
        return left_formula_robustness*right_formula_robustness

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

    def eval(self, traces):
        return Subtract(self.formulas[1], self.formulas[0]).eval(traces)

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

    def eval(self, traces):
        return np.abs(self.formulas[0].eval(traces))

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

    def eval(self, traces):
        return Or(Not(self.formulas[0]), self.formulas[1]).eval(traces)

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

    def eval(self, traces):
        robustness = self.formula_robustness.eval(traces)
        return np.where(robustness == 0, 1, robustness)	

class Not(STL):

    def __init__(self, formula):
        self.formulas = [formula]
        if self.formulas[0].range is None:
            self.range = None
        else:
            self.range = [-1*self.formulas[0].range[1], -1*self.formulas[0].range[0]]

        self.horizon = self.formulas[0].horizon

    def eval(self, traces):
        return -1 * self.formulas[0].eval(traces)

class Next(STL):

    def __init__(self, formula):
        self.formulas = [formula]
        self.range = self.formulas[0].range.copy() if self.formulas[0].range is not None else None
        self.horizon = 1 + self.formulas[0].horizon

    def eval(self, traces):
        formula_robustness = self.formulas[0].eval(traces)
        res = np.roll(formula_robustness, -1)
        return res[:-1]

class Global(STL):

    def __init__(self, lower_time_bound, upper_time_bound, formula):
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.formulas = [formula]
        self.range = None if self.formulas[0].range is None else self.formulas[0].range.copy()
        self.horizon = self.upper_time_bound + self.formulas[0].horizon

    def eval(self, traces):
        robustness = self.formulas[0].eval(traces)
        result = np.empty(shape=(len(robustness)))
        for current_time_pos in range(len(traces.timestamps) - 1, -1, -1):
            # Lower and upper times for the current time.
            lower_bound = traces.timestamps[current_time_pos] + self.lower_time_bound
            upper_bound = traces.timestamps[current_time_pos] + self.upper_time_bound

            # Find the corresponding positions in timestamps.
            lower_bound_pos = traces.search_time_index(lower_bound, start=current_time_pos)
            upper_bound_pos = traces.search_time_index(upper_bound, start=lower_bound_pos)

            # Find minimum between the positions.
            if lower_bound_pos == -1:
                r = float("inf")
            else:
                start_pos = lower_bound_pos
                end_pos = upper_bound_pos + 1 if upper_bound_pos >= 0 else len(traces.timestamps)
                r = np.min(robustness[start_pos: end_pos])

            result[current_time_pos] = r

        return result

class Finally(STL):

    def __init__(self, lower_time_bound, upper_time_bound, formula):
        self.upper_time_bound = upper_time_bound
        self.lower_time_bound = lower_time_bound
        self.formulas = [formula]
        self.formula_robustness = Not(Global(self.lower_time_bound, self.upper_time_bound, Not(self.formulas[0])))
        self.range = self.formula_robustness.range.copy() if self.formula_robustness.range is not None else None
        self.horizon = self.upper_time_bound + self.formulas[0].horizon

    def eval(self, traces):
        return self.formula_robustness.eval(traces)

class Or(STL):

    def __init__(self, *args, nu=None):
        self.formulas = list(args)
        # We save the actual definition in another attribute in order to work
        # correctly with nonassociativity and the parser.
        self.formula_robustness = Not(And(*[Not(f) for f in self.formulas]))
        self.range = self.formula_robustness.range.copy() if self.formula_robustness.range is not None else None
        self.horizon = self.formula_robustness.horizon

    def eval(self, traces):
        return self.formula_robustness.eval(traces)

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

    def eval(self, traces):
        if self.nu is None:
            return self._eval_traditional(traces)
        else:
            return self._eval_alternative(traces, self.nu)

    def _eval_traditional(self, traces):
        """This is the usual and."""

        # Evaluate the robustness of all subformulas and save the robustness
        # signals into one 2D array.
        M = len(self.formulas)
        for i in range(M):
            r = self.formulas[i].eval(traces)
            if i == 0:
                rho = np.empty(shape=(M, len(r)))
            rho[i,:] = r

        return np.min(rho, axis=0)

    def _eval_alternative(self, traces, nu):
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

