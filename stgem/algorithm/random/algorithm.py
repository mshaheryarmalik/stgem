#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib

import numpy as np

from stgem.algorithm import Algorithm
from stgem.algorithm.random.model import Random_Model

class Random(Algorithm):
    """
    Baseline random algorithm for generating a test suite.
    """

    def __init__(self, sut, test_repository, objective_funcs, objective_selector, parameters, logger=None):
        super().__init__(sut, test_repository, objective_funcs, objective_selector, parameters, logger)

        self.N_models = sum(f.dim for f in self.objective_funcs)
        module = importlib.import_module("stgem.algorithm.random.model")
        model = self.model if "model" in self.parameters else "Random_Model"
        self.model_class = getattr(module, model)
        self.models = [self.model_class(sut=self.sut, parameters=self.parameters, logger=logger) for _ in range(self.N_models)]

    def generate_test(self):
        self.perf.timer_start("total")

        # TODO: Implement the usage of predefined random data.

        while True:
            # Generate a new test.
            # -----------------------------------------------------------------------
            self.perf.timer_start("generation")
            self.log("Starting to generate test {}.".format(self.test_repository.tests + 1))
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
            self.perf.save_history("generation_time", self.perf.timer_reset("generation"))
            self.perf.save_history("N_tests_generated", rounds)
            self.perf.save_history("N_invalid_tests_generated", invalid)

            # Execute the test on the SUT.
            # -----------------------------------------------------------------------
            self.log("Chose test {} with predicted maximum objective 1. Generated total {} tests of which {} were invalid.".format(new_test, rounds, invalid))
            self.log("Executing the test...")

            sut_output = self.sut.execute_test(new_test)
            # Check if the SUT output is a vector or a signal.
            if np.isscalar(sut_output[0]):
                output = [self.objective_funcs[i](sut_output) for i in range(self.N_models)]
            else:
                output = [self.objective_funcs[i](**sut_output) for i in range(self.N_models)]

            self.log("The actual fitness {} for the generated test.".format(output))

            # Add the new test to the test suite.
            # -----------------------------------------------------------------------
            idx = self.test_repository.record(new_test.reshape(-1), output)
            self.objective_selector.update(np.argmax(output))

            self.perf.save_history("training_time", 0)

            self.perf.timers_hold()
            yield idx
            self.perf.timers_resume()

