#!/usr/bin/python3
# -*- coding: utf-8 -*-

import copy

from stgem.performance import PerformanceData

"""
Currently the use_previous_rng parameter is used so that the setup method can
be called several times without the RNG being advanced. This is especially
important with hyperparameter tuning as then setup is called when
hyperparameters are changed and if the setup involves setting up machine
learning models, the initial weights or other parameters can be completely
different.

It is up to the child class to implement RNG saving and restoration.
"""

class Model:
    """
    Base class for all models.
    """

    default_parameters = {}

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = {}

        # merge default_parameters and parameters, the later takes priority if a key appears in both dictionaries
        # the result is a new dictionary
        self.parameters = self.default_parameters | parameters

        self.previous_rng_state = None

    def setup(self, search_space, device, logger=None, use_previous_rng=False):
        if use_previous_rng and self.previous_rng_state is None:
            raise Exception("No previous RNG state to be used.")

        self.search_space = search_space
        self.parameters["input_dimension"] = self.search_space.input_dimension
        self.device = device
        self.logger = logger
        self.log = lambda msg: (self.logger("model", msg) if logger is not None else None)

        self.perf = PerformanceData()

    def __getattr__(self, name):
        if "parameters" in self.__dict__:
            if name in self.parameters:
                return self.parameters.get(name)

        raise AttributeError(name)

    def reset(self):
        pass

    def generate_test(self, N=1):
        """
        Generate N random tests.

        Args:
          N (int): Number of tests to be generated.

        Returns:
          output (np.ndarray): Array of shape (N, self.search_space.input_dimension).
        """

        raise NotImplementedError()

    def predict_objective(self, test):
        """
        Predicts the objective function value of the given tests.

        Args:
          test (np.ndarray): Array of shape (N, self.search_space.input_dimension).

        Returns:
          output (np.ndarray): Array of shape (N, 1).
        """

        raise NotImplementedError()

