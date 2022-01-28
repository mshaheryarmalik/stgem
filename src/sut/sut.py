#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Here is a base class implementation for systems under test (SUTs). We do not
strictly enforce the input and output representations for flexibility, but we
have the following conventions which should be followed if possible.

Inputs:
-------
We have two input formats: vectors and discrete signals. Notice that it should
always be possible to give several inputs at once since then parallelization
can be used.

Vectors inputs should be numpy arrays of floats. The SUT should allow the
execution of variable-length input vectors whenever this makes sense (e.g.,
when the input is interpretable as time series). Since we allow several inputs
to be specified at once, the input should here be an array of vectors.

Discrete signals.
"""

import numpy as np

class SUT:
  """
  Base class implementing a system under test.
  """

  def execute_test(self, test):
    raise NotImplementedError()

  def execute_random_test(self):
    raise NotImplementedError()

  def sample_input_space(self):
    raise NotImplementedError()

  def validity(self, test):
    """
    Basic validator which deems all tests valid.
    """

    return 1

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

