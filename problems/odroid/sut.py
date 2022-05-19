#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import numpy as np

from stgem.sut import SUT
from .util import generate_odroid_data

class OdroidSUT(SUT):
    """
    Implements the Odroid system under test.

    The input is the SUT configuration with dimension 6 and the output of a test
    is a 3-tuple (power, performance, efficiency) scaled to [0, 1].
    """

    def __init__(self, parameters):
        super().__init__(parameters)

        self.idim = 6
        self.odim = 3
        self.input_range = [[-1, 1] for _ in range(self.idim)]
        self.output_range = [[0, 1] for _ in range(self.odim)]

        self.ndimensions = None
        self.dataX = None
        self.dataY = None
        self.scaleX = None
        self.scaleY1 = None
        self.scaleY2 = None
        self.scaleY3 = None

        try:
            self._load_odroid_data()
        except:
            raise

    def _load_odroid_data(self):
        # Check if we have a npy file. Otherwise we attempt to generate such a file
        # from a csv file.
        if not os.path.exists(self.data_file):
            if not self.data_file.endswith(".npy"):
                raise Exception("The Odroid data file does not have extension .npy.")
            csv_file = self.data_file[:-4] + ".csv"
            if not os.path.exists(csv_file):
                raise Exception("No Odroid csv file '{}' available for data generation.".format(csv_file))
            generate_odroid_data(csv_file)

        data = np.load(self.data_file)

        # Set number of input dimensions.
        self.ndimensions = 6

        self.dataX = data[:, 0 : self.ndimensions]
        self.dataY = data[:, self.ndimensions :]

        # Normalize the inputs to [-1, 1].
        self.scaleX = self.dataX.max(axis=0)
        self.dataX = (self.dataX / self.scaleX) * 2 - 1
        # Normalize the outputs to [0, 1].
        self.scaleY1 = self.dataY[:, 0].max(axis=0)
        self.dataY[:, 0] = self.dataY[:, 0] / self.scaleY1
        self.scaleY2 = self.dataY[:, 1].max(axis=0)
        self.dataY[:, 1] = self.dataY[:, 1] / self.scaleY2
        self.scaleY3 = self.dataY[:, 2].max(axis=0)
        self.dataY[:, 2] = self.dataY[:, 2] / self.scaleY3

    def _execute_test(self, test):
        """
        Execute the given tests on the SUT. As not all possible parameters have a
        test value in the data, we find the closest test from the test data
        (Euclidean distance) and return its value.

        Args:
          test (np.ndarray): Array with shape (1,N) or (N) with
                             N = self.ndimensions of floats in [-1, 1].

        Returns:
          result (np.ndarray): Array of shape (3) of floats in [0, 1].
        """

        if not (test.shape == (1, self.ndimensions) or test.shape == (self.ndimensions,)):
            raise ValueError("Input array expected to have shape (1, {0}) or ({0}).".format(self.ndimensions))

        distances = np.sum((self.dataX - test)**2, axis=1)
        return self.dataY[np.argmin(distances)]

