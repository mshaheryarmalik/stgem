#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

from stgem.algorithm import Algorithm

class SimulatedAnnealing(Algorithm):
    """Implements simulated annealing."""

    default_parameters = {"radius": 0.01}
    
    def setup(self, search_space, device=None, logger=None):
        super().setup(search_space, device, logger)
        self.prec = 0
        self.rounds = 0

    def _temperature(self, budget_remaining):
        # This is in a way arbitrary, but seems to give reasonably small
        # probabilities when the diff is small (as we often have).
        return 0.5*budget_remaining**4

    def _probability(self, diff, temperature):
        # This is the standard Metropolis condition.
        return np.exp(-diff / temperature)

    def _neighbor(self, x):
        # Return a random displacement of x.
        neighbor = x + self.radius * np.random.uniform(-1, 1, size=len(x))
        np.clip(neighbor, -1, 1, out=neighbor)

        return neighbor

    def do_train(self, active_outputs, test_repository, budget_remaining):
        pass

    def do_generate_next_test(self, active_outputs, test_repository, budget_remaining):
        if test_repository.tests <= 1:
            # The first two tests are completely random.
            # TODO: Invalid tests for the first two rounds do not work exactly
            # nicely, but we ignore this for now.
            new_test = self.search_space.sample_input_space()
            if test_repository.tests == 0:
                self.candidate = new_test
                self.current = self.candidate
            else:
                v = test_repository.get(-1)[2]
                self.current_obj = min(v[i] for i in active_outputs)
                self.candidate = new_test
        else:
            # Find the minimum objective for the previously proposed test.
            v = test_repository.get(-1)[2]
            self.candidate_obj = min(v[i] for i in active_outputs)

            # If the new test is better, accept it as the new current test.
            # Otherwise accept it with a certain probability.
            diff = self.candidate_obj - self.current_obj
            if diff < 0 or np.random.random_sample() < self._probability(diff, self._temperature(budget_remaining)):
                self.current = self.candidate
                self.current_obj = self.candidate_obj

        invalid = 0
        while True:
            # Propose a neighbor of the current test.
            self.candidate = self._neighbor(self.current)

            if self.search_space.is_valid(self.candidate) == 1: break
            invalid += 1

        return self.candidate

