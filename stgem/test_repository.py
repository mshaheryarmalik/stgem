import time

import numpy as np

class TestRepository:

    def __init__(self):
        self._tests = []               # SUTInput objects.
        self._outputs = []             # SUTOutput objects.
        self._objectives = []          # Objectives for the SUTOutput.
        self._performance_records = [] # PerformanceRecord objects.

        self.unfinalized = False
        self.current_test = -1
        self.tests = 0
        self.minimum_objective = float("inf")

    @property
    def indices(self):
        return list(range(self.tests))

    def new_record(self):
        self._performance_records.append({})
        self.unfinalized = True
        self.current_test += 1
        return PerformanceRecordHandler(self._performance_records[-1])

    def record_input(self, sut_input):
        if not self.unfinalized: return
        if len(self._tests) <= self.current_test:
            self._tests.append(sut_input)
        else:
            self._tests[-1] = sut_input

    def record_output(self, sut_output):
        if not self.unfinalized: return
        if len(self._outputs) <= self.current_test:
            self._outputs.append(sut_output)
        else:
            self._outputs[-1] = sut_output

    def record_objectives(self, objectives):
        if not self.unfinalized: return
        if len(self._objectives) <= self.current_test:
            self._objectives.append(objectives)
        else:
            self._objectives[-1] = objectives

        # TODO: This does not work correctly if this method is called twice.
        # Save minimum objective component observed.
        m = min(objectives)
        if m < self.minimum_objective:
            self.minimum_objective = m

    def finalize_record(self):
        self.unfinalized = False
        self.tests += 1

        return self.current_test

    def get(self, *args, **kwargs):
        if len(args) == 0:
            # Return all tests.
            args = self.indices

        # TODO: Check bounds.

        if len(args) == 1:
            if isinstance(args[0], (int, np.integer)):
                # Return a single test.
                return self._tests[args[0]], self._outputs[args[0]], self._objectives[args[0]]
            else:
                args = args[0]

        # Return multiple tests.
        X = [self._tests[i] for i in args]
        Z = [self._outputs[i] for i in args]
        Y = [self._objectives[i] for i in args]

        return X, Z, Y

    def performance(self, test_idx):
        return PerformanceRecordHandler(self._performance_records[test_idx])

class PerformanceRecordHandler:

    def __init__(self, record):
        self._record = record
        self.timers = {}

    def timer_start(self, timer_id):
        if timer_id in self.timers and self.timers[timer_id] is not None:
            raise Exception("Restarting timer '{}' without resetting.".format(timer_id))

        self.timers[timer_id] = time.perf_counter()

    def timer_reset(self, timer_id):
        if not timer_id in self.timers:
            raise Exception("No timer '{}' to be reset.".format(timer_id))
        if self.timers[timer_id] is None:
            raise Exception("Timer '{}' already reset.".format(timer_id))

        time_elapsed = time.perf_counter() - self.timers[timer_id]
        self.timers[timer_id] = None

        return time_elapsed

    def timers_hold(self):
        for timer_id, t in self.timers.items():
            if t is not None:
                self.timers[timer_id] = time.perf_counter() - t

    def timers_resume(self):
        self.timers_hold()

    def obtain(self, performance_id):
        if not performance_id in self._record:
            raise Exception("No record with identifier '{}'.".format(performance_id))
        return self._record[performance_id]

    def record(self, performance_id, value):
        self._record[performance_id] = value

