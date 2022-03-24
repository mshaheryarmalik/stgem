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
            parameters = copy.deepcopy(self.default_parameters)
        self.parameters = parameters

    def setup(self, sut, device, logger=None):
        self.sut = sut
        self.device = device

        self.logger = logger
        self.log = lambda s: self.logger.model.info(s) if logger is not None else None

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
          output (np.ndarray): Array of shape (N, self.sut.ndimensions).
        """

        raise NotImplementedError()

    def predict_objective(self, test):
        """
        Predicts the objective function value of the given tests.

        Args:
          test (np.ndarray): Array of shape (N, self.sut.ndimensions).

        Returns:
          output (np.ndarray): Array of shape (N, 1).
        """

        raise NotImplementedError()

