#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

class SUT:
  """
  Base class implementing a system under test.
  """

  def __init__(self):
    self.ndimensions = None
    self.dataX = None
    self.dataY = None
    self.target = 1.0

  def distance(self, X, Y):
    """
    Returns the distance between two tests. This might not make sense for
    arbitrary system under test; we return Euclidean distance by default.

    Args:
      X (np.ndarray): Test array of shape (1, self.ndimensions) or (self.dimensions).
      Y (np.ndarray): Test array of shape (1, self.ndimensions) or (self.dimensions).

    Returns:
      result (float): The Euclidean distance of X and Y.
    """

    if len(X.shape) > 2 or len(Y.shape) > 2:
      raise ValueError("The tests must be 1- or 2-dimensional arrays.")
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    if X.shape[0] != Y.shape[0]:
      raise ValueError("The tests must have the same dimension.")

    return np.linalg.norm(X - Y)

  def execute_test(self, tests):
    raise NotImplementedError()

  def execute_random_test(self, N=1):
    raise NotImplementedError()

  def sample_input_space(self, N=1):
    raise NotImplementedError()

def SUT_MO_W(SUT):
  """
  A class which combines two or several systems under test for multiobjective
  optimization using scalarization with weights.
  """

  def __init__(self, suts, weights):
    self.suts = suts
    self.ndimensions = self.suts[0].ndimensions

    # Test that all SUTs accept tests with same input dimension.
    for sut in self.suts[1:]:
      if sut.ndimensions != self.ndimensions:
        raise ValueError("All systems under test should accept tests with the same input dimension.")

    # Normalize the weights.
    self.sut_weights = np.array(weights)
    self.sut_weights = self.sut_weights / np.sum(self.sut_weights)

  def execute_test(self, tests):
    """
    Execute the given tests on the SUT.

    Args:
      tests (np.ndarray): Array of N tests with shape (N, self.ndimensions).

    Returns:
      result (np.ndarray): Array of shape (N, 1).
    """

    if len(tests.shape) != 2 or tests.shape[1] != self.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.ndimensions))

    result = np.zeros(shape=(tests.shape[0], 1))
    for n, test in enumerate(tests):
      sut_results = [sut.execute_test(test[n]) for sut in self.suts]
      result[n,0] = weight*np.array(sut_results)

    return result

  def execute_random_test(self, N=1):
    """
    Execute N random tests and return their outputs.

    Args:
      N (int): The number of tests to be executed.

    Returns:
      tests (np.ndarray):   Array of shape (N, self.ndimensions).
      outputs (np.ndarray): Array of shape (N, 1).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")

    dataX = self.sample_input_space(N)
    dataY = self.execute_test(dataX)
    return dataX, dataY

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

    return self.suts[0].sample_input_space(N)
