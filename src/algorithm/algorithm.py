#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time

import numpy as np

class Algorithm:
    """
    Base class for all test suite generation algorithms.
    """

    def __init__(self, sut, test_repository, objective_func, objective_selector, logger=None):
        self.sut = sut
        self.test_repository = test_repository
        self.objective_func = objective_func
        self.objective_selector = objective_selector

        self.logger = logger
        self.log = (lambda s: self.logger.algorithm.info(s) if logger is not None else None)

        self.test_suite = []
        self.timers = {}
        self.histories = {}

    def __getattr__(self, name):
        value = self.parameters.get(name)
        if value is None:
            raise AttributeError(name)

        return value

    def get_history(self, id):
        if not id in self.timers:
            raise Exception("No history for the identifier '{}'.".format(id))
        return self.histories[id]

    def save_history(self, id, value, single=False):
        if not single:
            if not id in self.timers:
                self.histories[id] = []
            self.histories[id].append(value)
        else:
            self.histories[id] = value

    def timer_start(self, id):
        # TODO: Implement a good time for all platforms.
        if id in self.timers and self.timers[id] is not None:
            raise Exception("Restarting timer '{}' without resetting.".format(id))

        self.timers[id] = time.monotonic()

    def timer_reset(self, id):
        if not id in self.timers:
            raise Exception("No timer '{}' to be reset.".format(id))
        if self.timers[id] is None:
            raise Exception("Timer '{}' already reset.".format(id))

        time_elapsed = time.monotonic() - self.timers[id]
        self.timers[id] = None

        return time_elapsed

    def timers_hold(self):
        for id, t in self.timers.items():
            if t is not None:
                self.timers[id] = time.monotonic() - self.timers[id]

    def timers_resume(self):
        self.timers_hold()

    def generate_test(self):
        raise NotImplementedError()

    def _generate_initial_random_tests(self):
        tests_generated = 0
        if self.use_predefined_random_data:
          self.log("Loading {} predefined random tests.".format(len(self.predefined_random_data_idx)))
          # We need to return each predefined test, update the objective
          # selector, and add (fake) times.
          for idx in self.predefined_random_data_idx:
            self.timer_start("generation")
            self.test_suite.append(idx)
            self.objective_selector.update(np.argmax(self.test_repository.get(idx)[1]))
            tests_generated += 1
            self.save_history("generation_time", self.timer_reset("generation"))
            self.save_history("N_tests_generated", 1)
            self.save_history("N_invalid_tests_generated", 0)
            self.save_history("execution_time", 0)

            self.timers_hold()
            yield idx
            self.timers_resume()

        if tests_generated < self.N_random_init:
          self.timer_start("generation")
          self.log("Generating and running {} random valid tests.".format(self.N_random_init - tests_generated))
          invalid = 0
          while tests_generated < self.N_random_init:
            # TODO: Same test could be generated twice, but this is unlikely.
            test = self.sut.sample_input_space()
            if self.sut.validity(test) == 0:
              invalid += 1
              continue
            tests_generated += 1

            # Save information on how many tests needed to be generated etc.
            self.save_history("generation_time", self.timer_reset("generation"))
            self.save_history("N_tests_generated", invalid + 1)
            self.save_history("N_invalid_tests_generated", invalid)

            self.log("Executing {} ({}/{})".format(test, tests_generated, self.N_random_init))
            self.timer_start("execution")
            test_output = self.objective_func(self.sut.execute_test(test))
            self.save_history("execution_time", self.timer_reset("execution"))

            idx = self.test_repository.record(test, test_output)
            self.test_suite.append(idx)
            self.objective_selector.update(np.argmax(test_output))

            self.log("Result: {}".format(test_output))

            self.timers_hold()
            yield idx
            self.timers_resume()

            self.timer_start("generation")
            invalid = 0

        self.timer_reset("generation")

