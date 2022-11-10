import math, time

from stgem.sut import SUTOutput

class Budget:

    def __init__(self):
        # Setup the quantities and their initial values.
        default_quantities = ["executions", "execution_time", "generation_time", "training_time", "wall_time"]
        self.quantities = {q:0 for q in default_quantities}

        # Setup a budget directly reporting the corresponding quantity for the
        # default budgets.
        def make_func(q):
            # This is to work around Python variable scoping.
            return lambda quantities: quantities[q]
        self.budgets = {q:make_func(q) for q in self.quantities}
        # Setup wall time reporting.
        self.budgets["wall_time"] = lambda quantities: time.perf_counter() - self.initial_wall_time

        # Setup default budgets to be from 0 to infinity.
        self.budget_ranges = {b:[0,math.inf] for b in self.budgets}

        # We start wall time counting only when the thresholds have been
        # updated for the first time.
        self.initial_wall_time = -1

    def update_threshold(self, budget_threshold):
        # We set up the wall time counter here if it has not been started yet.
        if self.initial_wall_time < 0:
            self.initial_wall_time = time.perf_counter()

        # Use specified values; infinite budget for nonspecified quantities.
        for name in budget_threshold:
            if budget_threshold[name] < self.budgets[name](self.quantities):
                raise Exception("Cannot update budget threshold '{}' to '{}' since its below the already consumed budget '{}'.".format(name, quantity_threshold[name], self.budgets[name](self.quantities)))

            if self.budget_ranges[name][1] < budget_threshold[name]:
                self.budget_ranges[name][0] = self.budget_ranges[name][1]
                self.budget_ranges[name][1] = budget_threshold[name]
            else:
                self.budget_ranges[name][1] = budget_threshold[name]

    def remaining(self):
        """Return the minimum amount of budget left among all budget as a
        number in [0,1]."""

        return min(self.used().values())

    def used(self):
        """Return for each budget a number in [0,1] indicating how much budget
        is left."""

        result = {}
        for name in self.budgets:
            start = self.budget_ranges[name][0]
            end = self.budget_ranges[name][1]
            value = self.budgets[name](self.quantities)
            if value >= start:
                remaining = 1.0 - (value - start) / (end - start)
            else:
                remaining = 1.0

            result[name] = remaining

        return result

    def _consume(self, quantity, value=1):
        if quantity in self.quantities:
            self.quantities[quantity] += value

    def _consume_on_output(self, output):
        """Consume a budget based on an SUTOutput object. By default this does
        nothing, but a child class can define the desired behavior."""

        pass

    def consume(self, *args, **kwargs):
        """Consume a budget based on a SUTOutput or a budget id and an associated value (by default 1).

        If the budget id is not recognized, nothing is done."""

        if isinstance(args[0], SUTOutput):
            self._consume_on_output(args[0])
        else:
            self._consume(args[0], args[1] if len(args) > 1 else 1)

