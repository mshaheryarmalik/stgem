#!/usr/bin/python3
# -*- coding: utf-8 -*-

class TestRepository:
    def __init__(self):
        self._sut_tests = []           # Inputs as expected by the SUT
        self._normalized_tests = []    # Normalized inputs as expected by the search and generation algorithms
        self._sut_outputs = []         # Outputs as produced by the SUT
        self._normalized_outputs = []  # Normalized outputs as expected by the search and generation algorithms
        self.minimum_normalized_output = float("inf")

    @property
    def tests(self):
        return len(self._normalized_tests)

    @property
    def indices(self):
        return list(range(len(self._normalized_tests)))

    def record(self, sut_test, normalized_test, sut_output, normalized_output):
        self._sut_tests.append(sut_test)
        self._sut_outputs.append(sut_output)
        self._normalized_tests.append(normalized_test)
        self._normalized_outputs.append(normalized_output)

        # Save minimum objective component observed.
        m = min(normalized_output)
        if m < self.minimum_normalized_output:
            self.minimum_normalized_output = m

        return len(self._normalized_tests) - 1

    def get(self, *args, **kwargs):
        if len(args) == 0:
            # Return all tests.
            return self._sut_tests, self._normalized_tests, self._sut_outputs, self._normalized_outputs

        if len(args) == 1:
            if isinstance(args[0], int):
                # Return a single test.
                return self._sut_tests[args[0]], self._normalized_tests[args[0]], self._sut_outputs[args[0]], self._normalized_outputs[args[0]]
            else:
                args = args[0]

        # Return multiple tests.
        W = [self._sut_tests[i] for i in args]
        X = [self._normalized_tests[i] for i in args]
        Y = [self._sut_outputs[i] for i in args]
        Z = [self._normalized_outputs[i] for i in args]

        return W, X, Y, Z

