#!/usr/bin/python3
# -*- coding: utf-8 -*-

import copy

from stgem.performance import PerformanceData

class Model:
    """
    Base class for all models.
    """

    default_parameters = {}

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = {}

        # merge deafult_parameters and parameters, the later takes priority if a key appears in both dictionaries
        # the result is a new dictionary
        self.parameters = self.default_parameters | parameters

    def setup(self, search_space, device, logger=None):
        self.search_space = search_space
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

    def load_from_file(self,fn):
        raise NotImplementedError()

    def save_to_file(self,fn):
        raise NotImplementedError()

    def get_input_dimension(self):
        raise NotImplementedError()