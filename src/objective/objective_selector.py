#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np


class ObjectiveSelector:
    def __init__(self, N_objectives,parameters):
        self.dim = N_objectives
        self.parameters=parameters

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

    def select(self):
        return self.select_all()


class ObjectiveSelectorMAB(ObjectiveSelector):
    """
    Simple multi-armed bandit inspired model selector which uses the success
    frequency of obtaining the best objective component to randomly select one
    model. A warm-up period can be defined where all models are returned.
    """

    def __init__(self, N_objectives, parameters):
        super().__init__(N_objectives,parameters)
        self.warm_up = self.parameters["warm_up"]
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
