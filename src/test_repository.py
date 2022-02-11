#!/usr/bin/python3
# -*- coding: utf-8 -*-


class TestRepository:
    def __init__(self):
        self.tests = []
        self.outputs = []

    def record(self, test, output):
        self.tests.append(test)
        self.outputs.append(output)

        return len(self.tests) - 1

    def get(self, idx):
        if isinstance(idx, int):
            return self.tests[idx], self.outputs[idx]
        else:
            X = [self.tests[i] for i in idx]
            Y = [self.outputs[i] for i in idx]
            return X, Y

