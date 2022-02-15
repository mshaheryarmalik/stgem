#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time

import numpy as np

from performance import PerformanceData

class Algorithm:
    """
    Base class for all test suite generation algorithms.
    """

    def __init__(self, sut, test_repository, objective_funcs, objective_selector, logger=None):
        self.sut = sut
        self.test_repository = test_repository
        self.objective_funcs = objective_funcs
        self.objective_selector = objective_selector

        self.logger = logger
        self.log = (lambda s: self.logger.algorithm.info(s) if logger is not None else None)

        self.test_suite = []

        self.perf = PerformanceData()

    def __getattr__(self, name):
        value = self.parameters.get(name)
        if value is None:
            raise AttributeError(name)

        return value

    def generate_test(self):
        raise NotImplementedError()

    def _generate_initial_random_tests(self):
        tests_generated = 0
        # It is up to the caller to ensure that the initial random data is good
        # (does not have the same test twice, all tests are valid, etc.).
        if self.use_predefined_random_data:
          self.log("Loading {} predefined random tests.".format(len(self.predefined_random_data_idx)))
          # We need to return each predefined test, update the objective
          # selector, and add (fake) times.
          for idx in self.predefined_random_data_idx:
            self.perf.timer_start("generation")
            self.test_suite.append(idx)
            self.objective_selector.update(np.argmax(self.test_repository.get(idx)[1]))
            tests_generated += 1
            self.perf.save_history("generation_time", self.perf.timer_reset("generation"))
            self.perf.save_history("N_tests_generated", 1)
            self.perf.save_history("N_invalid_tests_generated", 0)
            self.perf.save_history("execution_time", 0)

            self.perf.timers_hold()
            yield idx
            self.perf.timers_resume()

        if tests_generated < self.N_random_init:
          self.perf.timer_start("generation")
          self.log("Generating and running {} random valid tests.".format(self.N_random_init - tests_generated))
          invalid = 0
          distance_invalid = 0
          while tests_generated < self.N_random_init:
            test = self.sut.sample_input_space()

            # Select only valid tests.
            if self.sut.validity(test) == 0:
              invalid += 1
              continue

            # Select sufficiently different tests if desired.
            if len(self.test_suite) > 0 and "random_search_min_distance" in self.parameters and self.random_search_min_distance > 0:
                d = self.sut.min_distance(np.asarray(self.test_repository.get(self.test_suite)[0]).reshape(len(self.test_suite), -1), test)
                if d < self.random_search_min_distance:
                    distance_invalid += 1
                    continue

            tests_generated += 1

            # Save information on how many tests needed to be generated etc.
            self.perf.save_history("generation_time", self.perf.timer_reset("generation"))
            self.perf.save_history("N_tests_generated", invalid + distance_invalid + 1)
            self.perf.save_history("N_invalid_tests_generated", invalid)

            self.log("Executing {} ({}/{})".format(test, tests_generated, self.N_random_init))

            sut_output = self.sut.execute_test(test)
            # Check if the SUT output is a vector or a signal.
            if np.isscalar(sut_output[0]):
                test_output = [self.objective_funcs[i](sut_output) for i in range(self.N_models)]
            else:
                test_output = [self.objective_funcs[i](**sut_output) for i in range(self.N_models)]

            idx = self.test_repository.record(test.reshape(-1), test_output)
            self.test_suite.append(idx)
            self.objective_selector.update(np.argmin(test_output))

            self.log("Result: {}".format(test_output))

            self.perf.timers_hold()
            yield idx
            self.perf.timers_resume()

            self.perf.timer_start("generation")
            invalid = 0

        self.perf.timer_reset("generation")

