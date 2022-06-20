#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

class TestRepository:
    def __init__(self):
        self._tests = []      # SUTInput objects.
        self._outputs = []    # SUTOutput objects.
        self._objectives = [] # Objectives for the SUTOutput.
        self.minimum_objective = float("inf")

    @property
    def tests(self):
        return len(self._objectives)

    @property
    def indices(self):
        return list(range(len(self._objectives)))

    def record(self, sut_input, sut_output, objectives):
        self._tests.append(sut_input)
        self._outputs.append(sut_output)
        self._objectives.append(objectives)

        # Save minimum objective component observed.
        m = min(objectives)
        if m < self.minimum_objective:
            self.minimum_objective = m

        return len(self._objectives) - 1

    def get(self, *args, **kwargs):
        if len(args) == 0:
            # Return all tests.
            return self._tests, self._outputs, self._objectives

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

