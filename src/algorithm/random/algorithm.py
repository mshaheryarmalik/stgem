#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from algorithm.algorithm import Algorithm
from .model import Random_Model


class Random(Algorithm):
    """
    Baseline random algorithm for generating a test suite.
    """

    def __init__(self, sut, test_repository, objective_func, objective_selector, parameters, logger=None):
        super().__init__(sut, test_repository, objective_func, objective_selector, logger)
        self.parameters = parameters

        self.models = [Random_Model(self.sut, self.parameters, self.logger) for _ in range(self.objective_func.dim)]

    def generate_test(self):
        self.timer_start("total")

        # TODO: Implement the usage of predefined random data.

        while True:
            # Generate a new test.
            # -----------------------------------------------------------------------
            self.timer_start("generation")
            self.log("Starting to generate test {}.".format(len(self.test_suite) + 1))
            rounds = 0
            invalid = 0
            # Select a model randomly and generate a random valid test for it.
            m = np.random.choice(self.objective_selector.select())
            while True:
                rounds += 1
                new_test = self.models[m].generate_test()
                if self.sut.validity(new_test) == 0:
                    invalid += 1
                    continue

                break

            # Save information on how many tests needed to be generated etc.
            # -----------------------------------------------------------------------
            self.save_history("generation_time", self.timer_reset("generation"))
            self.save_history("N_tests_generated", rounds)
            self.save_history("N_invalid_tests_generated", invalid)

            # Execute the test on the SUT.
            # -----------------------------------------------------------------------
            self.log("Chose test {} with predicted maximum objective 1. Generated total {} tests of which {} were invalid.".format(new_test, rounds, invalid))
            self.log("Executing the test...")

            self.timer_start("execution")
            sut_output = self.sut.execute_test(new_test)
            self.save_history("execution_time", self.timer_reset("execution"))
            # Check if we get a vector of floats or a 2-tuple of arrays (signals).
            if np.isscalar(sut_output[0]):
                output = self.objective_func(sut_output)
            else:
                output = self.objective_func(*sut_output)

            self.log("The actual fitness {} for the generated test.".format(output))

            # Add the new test to the test suite.
            # -----------------------------------------------------------------------
            idx = self.test_repository.record(new_test, output)
            self.test_suite.append(idx)
            self.objective_selector.update(np.argmax(output))

            self.save_history("training_time", 0)

            self.timers_hold()
            yield idx
            self.timers_resume()
