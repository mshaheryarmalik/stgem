#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


class ObjectiveSelector:
    def __init__(self, objective_func):
        self.dim = objective_func.dim

    def select_all(self):
        return list(range(self.dim))

    def select(self):
        raise NotImplementedError()

    def update(self, idx):
        return


class ObjectiveSelectorAll(ObjectiveSelector):
    """
    Model selector which ignores everything an just returns all models.
    """

    def __init__(self, objective_func):
        super().__init__(objective_func)

    def select(self):
        return self.select_all()


class ObjectiveSelectorMAB(ObjectiveSelector):
    """
    Simple multi-armed bandit inspired model selector which uses the success
    frequency of obtaining the best objective component to randomly select one
    model. A warm-up period can be defined where all models are returned.
    """

    def __init__(self, objective_func, warm_up=0):
        super().__init__(objective_func)
        self.warm_up = warm_up
        self.total_calls = 0
        self.model_successes = [0 for i in range(self.dim)]

    def select(self):
        if self.total_calls <= self.warm_up:
            return self.select_all()
        else:
            p = [s / self.total_calls for s in self.model_successes]
            return [np.random.choice(range(0, self.dim), p=p)]

    def update(self, idx):
        try:
            self.model_successes[idx] += 1
        except IndexError:
            raise Exception("No model with index {}.".format(idx))
        self.total_calls += 1
