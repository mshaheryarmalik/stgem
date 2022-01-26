#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn

class Validator:
  """
  Base class implementing a validator for a system under test.
  """

  def __init__(self, input_size, validator_bb):
    """
    Initialize the class.

    Args:
      input_size (int):        Input space dimension.
      validator_bb (function): A black box function which takes a test as an
                               input and returns 0 (invalid) or 1 (valid).
    """

    # TODO: checks for arguments

    self.ndimensions = input_size
    self.validator_bb = validator_bb

  def validity(self, tests):
    """
    Validate the given test using the true validator.

    Args:
      tests (np.ndarray): Array of N tests with shape (N, self.ndimensions).

    Returns:
      result (np.ndarray): Array of shape (N, 1).
    """

    if len(tests.shape) != 2 or tests.shape[1] != self.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.ndimensions))

    result = np.zeros(shape=(tests.shape[0], 1))
    for n, test in enumerate(tests):
      result[n,0] = self.validator_bb(test)

    return result

