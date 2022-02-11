#!/usr/bin/python3
# -*- coding: utf-8 -*-

class Model:
    """
    Base class for all models.
    """

    def __init__(self, sut, parameters, logger=None):
        self.sut = sut
        self.parameters = parameters
        self.logger = logger
        self.log = lambda s: self.logger.model.info(s) if logger is not None else None

    def __getattr__(self, name):
        value = self.parameters.get(name)
        if value is None:
            raise AttributeError(name)

        return value

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

