#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import numpy as np

from config import *
import preprocess_data

class SUT:
  """
  Base class implementing a system under test.
  """

  def __init__(self):
    self.ndimensions = None
    self.dataX = None
    self.dataY = None

  def execute_test(self, parameters):
    raise NotImplementedError()

  def execute_random_tests(self, N=1):
    raise NotImplementedError()

  def sample_input_space(self, N=1):
    raise NotImplementedError()

class OdroidSUT(SUT):
  """
  Implements the Odroid system under test.
  """

  def __init__(self, output, target):
    """
    Initialize the class.

    Args:
      output (int):   Which output data is used (1 = power, 2 = performance,
                      3 = efficiency).
      target (float): Threshold for a positive test value.
    """

    super().__init__()
    self.output = output
    self.target = target

    self.ndimensions = None
    self.dataX = None
    self.dataY = None
    self.scaleX = None
    self.scaleY = None

    try:
      self._load_odroid_data(output, target)
    except:
      raise

  def _load_odroid_data(self, output, target):
    if not (1 <= output <= 3):
      raise Exception("Argument 'output' should be 1, 2 or 3.")

    file_name = config["odroid_file_base"] + ".npy"

    if not os.path.exists(file_name):
      preprocess_data.generate_odroid_data()

    data = np.load(file_name)

    # Set number of input dimensions.
    self.ndimensions = 6

    self.dataX = data[: , 0:self.ndimensions]
    self.dataY = data[: , self.ndimensions + output - 1]

    # Compute the number of positive tests in the data.
    self.total = self.dataY.shape[0]
    self.totalp = self.dataY[self.dataY >= self.target].shape[0]

    # Normalize the input to [-1, 1].
    self.scaleX = self.dataX.max(axis=0)
    self.dataX = (self.dataX / self.scaleX)*2 - 1
    # Normalize the output to [0, 1] (in order to use certain loss functions).
    self.scaleY = self.dataY.max(axis=0)
    self.dataY = self.dataY / self.scaleY
    # Scale the positive threshold as well.
    self.target = self.target / self.scaleY

  def execute_test(self, tests):
    """
    Execute the given tests on the SUT. As not all possible parameters have a
    test value in the data, we find the closest test from the test data
    (Euclidean distance) and return its value.

    Args:
      tests (np.ndarray): Array of N tests with shape (N, self.ndimensions).

    Returns:
      result (np.ndarray): Array of shape (N, 1).
    """

    if len(tests.shape) != 2 or tests.shape[1] != self.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.ndimensions))

    result = np.zeros(shape=(tests.shape[0], 1))
    for n in range(tests.shape[0]):
      distances = np.sum((self.dataX - tests[n,:])**2, axis=1)
      result[n,0] = self.dataY[np.argmin(distances)]

    return result

  def execute_random_tests(self, N=1):
    """
    Execute n random tests and return their outputs.

    Args:
      N (int): The number of tests to be executed.

    Returns:
      tests (np.ndarray):   Array of shape (N, self.ndimensions).
      outputs (np.ndarray): Array of shape (N, 1).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")

    # We simply sample n tests from the data and return the corresponding
    # outputs.
    # TODO: what to do if n > self.dataX.shape[0]?
    sampleX, _, sampleY, _ = train_test_split(self.dataX, self.dataY,
                                              train_size=N, test_size=1)
    return sampleX, sampleY.reshape(sampleX.shape[0], 1)

  def sample_input_space(self, N=1):
    """
    Return n samples (tests) from the input space.

    Args:
      N (int): The number of tests to be sampled.

    Returns:
      tests (np.ndarray): Array of shape (N, self.ndimensions).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")

    # We simply sample n tests from the data.
    # TODO: what to do if n > self.dataX.shape[0]?
    sampleX, _, _, _ = train_test_split(self.dataX, self.dataY, train_size=N,
                                        test_size=1)
    return sampleX

