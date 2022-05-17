#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from stgem.algorithm import Algorithm

class Random(Algorithm):
    """
    Baseline random algorithm for generating a test suite.
    """

    def do_train(self, active_outputs, test_repository):
        self.active_outputs= active_outputs
        self.test_repository= test_repository

    def do_generate_next_test(self):

        self.log("Starting to generate test {}.".format(self.test_repository.tests + 1))
        rounds = 0
        invalid = 0
        # Select a model randomly and generate a random valid test for it.
        # TODO: I don't understand this: m = np.random.choice(self.objective_selector.select())
        m = int(np.random.uniform(0, len(self.models)))

        while True:
            rounds += 1
            new_test = self.models[m].generate_test()
            assert new_test
            if self.search_space.is_valid(new_test) == 0:
                invalid += 1
                continue

            break

        # Save information on how many tests needed to be generated etc.
        # -----------------------------------------------------------------

        self.perf.save_history("N_tests_generated", rounds)
        self.perf.save_history("N_invalid_tests_generated", invalid)

        return new_test
