#!/usr/bin/python3
# -*- coding: utf-8 -*-


class TestRepository:
    def __init__(self):
        self._tests = []
        self._outputs = []

    @property
    def tests(self):
        return len(self._tests)

    @property
    def indices(self):
        return list(range(len(self._tests)))

    def record(self, test, output):
        self._tests.append(test)
        self._outputs.append(output)

        return len(self._tests) - 1

    def get(self, *args, **kwargs):
        if len(args) == 0:
            # Return all tests.
            return self._tests, self._outputs

        if len(args) == 1:
            if isinstance(args[0], int):
                # Return a single test.
                return self._tests[args[0]], self._outputs[args[0]]
            else:
                args = args[0]

        # Return multiple tests.
        X = [self._tests[i] for i in args]
        Y = [self._outputs[i] for i in args]

        return X, Y

