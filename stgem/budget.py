import math, time

class Budget:

    def __init__(self):
        self.budget_ids = ["executions", "execution_time", "generation_time", "training_time", "wall_time"]
        self.budget_values = {name:0 for name in self.budget_ids}
        self.budget_start_values = {name:0 for name in self.budget_ids}
        self.budget_end_values = {name:math.inf for name in self.budget_ids}
        # We start wall time counting only when the thresholds have been
        # updated for the first time.
        self.initial_wall_time = -1

    def update_threshold(self, budget_threshold):
        # We setup the wall time counter here if it has not been started yet.
        if self.initial_wall_time < 0:
            self.initial_wall_time = time.perf_counter()

        # Use specified values, infinite budget for nonspecified budgets.
        for name in budget_threshold:
            if budget_threshold[name] < self.budget_values[name]:
                raise Exception("Cannot update budget threshold '{}' to '{}' since its below the already consumed budget '{}'.".format(name, budget_threshold[name], self.budget_values[name]))

            if self.budget_end_values[name] < budget_threshold[name]:
                self.budget_start_values[name] = self.budget_end_values[name]
                self.budget_end_values[name] = budget_threshold[name]
            else:
                self.budget_end_values[name] = budget_threshold[name]

    def remaining(self):
        return min(self.used().values())

    def used(self):
        result = {}
        for name in self.budget_ids:
            start = self.budget_start_values[name]
            value = self.budget_values[name] if name != "wall_time" else time.perf_counter() - self.initial_wall_time
            end = self.budget_end_values[name]
            if value >= self.budget_start_values[name]:
                left = 1.0 - (value - start) / (end - start)
            else:
                left = 1.0

            result[name] = left

        return result

    def consume(self, budget_id, value=1):
        """Consume a budget, i.e., add the given value (default 1) to the used
        budget corresponding to the budget id.

        If the budget id is not recognized, nothing is done."""

        if budget_id == "wall_time":
            raise Exception("The wall time budget cannot be consumed.")

        if budget_id in self.budget_ids:
            self.budget_values[budget_id] += value

