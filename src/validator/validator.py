#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn

from neural_networks.validator import ValidatorNetwork

class Validator:
  """
  Base class implementing a validator for a system under test.
  """

  def __init__(self, sut, validator_bb, device):
    """
    Initialize the class.

    Args:
      sut (SUT):               System under test.
      validator_bb (function): A black box function which takes a test (for the
                               SUT) as an input and returns 0 (invalid) or 1
                               (valid).
      device (str):            "cpu" or "cuda"
    """

    # The system under test.
    self.sut = sut
    # This is the validator black box function.
    self.validator_bb = validator_bb
    self.device = device

    # This is the learned proxy for the black box function.
    self.modelV = ValidatorNetwork(self.sut.ndimensions, 64)
    # Loss functions.
    #self.loss = nn.BCELoss() # binary cross entropy
    self.loss = nn.MSELoss() # mean squared error
    # Optimizers.
    lr = 0.001
    self.optimizerV = torch.optim.Adam(self.modelV.parameters(), lr=lr)

  def validity(self, tests):
    """
    Validate the given test using the true validator.

    Args:
      tests (np.ndarray): Array of N tests with shape (N, self.sut.ndimensions).

    Returns:
      result (np.ndarray): Array of shape (N, 1).
    """

    if len(tests.shape) != 2 or tests.shape[1] != self.sut.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.sut.ndimensions))

    result = np.zeros(shape=(tests.shape[0], 1))
    for n, test in enumerate(tests):
      result[n,0] = self.validator_bb(test)

    return result

  def predict_validity(self, tests):
    """
    Validate the given test using the learned proxy validator.

    Args:
      tests (np.ndarray): Array of N tests with shape (N, self.sut.ndimensions).

    Returns:
      result (np.ndarray): Array of shape (N, 1).
    """

    if len(tests.shape) != 2 or tests.shape[1] != self.sut.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.sut.ndimensions))

    return self.modelV(torch.from_numpy(tests).float().to(self.device)).detach().numpy()

  def train(self, N):
    """
    Train a proxy for the validator function using N randomly sampled tests.

    Args:
      N (int): Number of tests to be sampled.
    """

    if N <= 0:
      raise ValueError("The number of tests to be sampled must be positive.")

    # Reset the proxy neural network.
    self.modelV = ValidatorNetwork(self.sut.ndimensions, 128)

    dataX = torch.from_numpy(self.sut.sample_input_space(N)).float().to(self.device)
    dataY = torch.from_numpy(self.validate(dataX)).float().to(self.device)
    self.train_with_batch(dataX, dataY, epochs=30)

  def train_with_batch(self, dataX, dataY, epochs=1):
    """
    Train the proxy for the validator using a batch of training data.

    Args:
      dataX (np.ndarray): Array of tests of shape (N, self.sut.ndimensions).
      dataY (np.ndarray): Array of test outputs of shape (N, 1).
      epochs (int):       Number of epochs (total training over the complete
                          data).
    """

    if len(dataX.shape) != 2 or dataX.shape[1] != self.sut.ndimensions:
      raise ValueError("Test array expected to have shape (N, {}).".format(self.sut.ndimensions))
    if len(dataY.shape) != 2 or dataY.shape[0] < dataX.shape[0]:
      raise ValueError("Output array should have at least as many elements as there are tests.")
    if epochs <= 0:
      raise ValueError("The number of epochs should be positive.")

    dataX = torch.from_numpy(dataX).float().to(self.device)
    dataY = torch.from_numpy(dataY).float().to(self.device)
    for n in range(epochs):
      output = self.modelV(dataX)
      V_loss = self.loss(output, dataY)
      self.optimizerV.zero_grad()
      V_loss.backward()
      self.optimizerV.step()

      acc = ((output > 0.5) == dataY).float().sum() / dataY.shape[0]
      #print("Epoch {}/{}, Loss: {}, Accuracy: {}".format(n+1, epochs, V_loss, acc))

