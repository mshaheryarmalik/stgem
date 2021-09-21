#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn

# For visualizing the computational graphs.
#from torchviz import make_dot

from neural_networks import GeneratorNetwork, DiscriminatorNetwork

class Model:
  """
  Base class for all models.
  """

  def __init__(self):
    pass

  def train_with_batch(self, dataX, dataY, epochs=1):
    raise NotImplementedError()

  def generate_test(self, N=1):
    raise NotImplementedError()

class GAN(Model):
  """
  Implements the GAN model.
  """

  def __init__(self, sut, device):
    super().__init__()

    self.sut = sut
    self.device = device

    self.modelG = None
    self.modelD = None
    # Input dimension for the noise inputted to the generator.
    self.noise_dim = 500
    # Number of neurons per layer in the neural networks.
    self.neurons = 128

    # Initialize neural network models.
    self.modelG = GeneratorNetwork(input_shape=self.noise_dim, output_shape=self.sut.ndimensions, neurons=self.neurons).to(self.device)
    self.modelD = DiscriminatorNetwork(input_shape=self.sut.ndimensions, neurons=self.neurons).to(self.device)

    # Loss functions.
    # TODO: figure out a reasonable default and make configurable.
    self.loss = nn.BCELoss() # binary cross entropy

    # Optimizers.
    # TODO: figure out reasonable defaults and make configurable.
    lr = 0.001
    self.optimizerD = torch.optim.Adam(self.modelD.parameters(), lr=lr)
    self.optimizerG = torch.optim.Adam(self.modelG.parameters(), lr=lr)

  def train_with_batch(self, dataX, dataY, epochs=1):
    """
    Train the GAN with a new batch of learning data.

    Args:
      dataX (np.ndarray): Array of tests of shape (N, self.sut.ndimensions).
      dataY (np.ndarray): Array of test outputs of shape (N, 1).
      epochs (int):       The number of epochs.
    """

    if len(dataX.shape) != 2 or dataX.shape[1] != self.sut.ndimensions:
      raise ValueError("Test array expected to have shape (N, {}).".format(self.ndimensions))
    if len(dataY.shape) != 2 or dataY.shape[0] < dataX.shape[0]:
      raise ValueError("Output array should have at least as many elements as there are tests.")
    if epochs <= 0:
      raise ValueError("The number of epochs should be positive.")

    dataX = torch.from_numpy(dataX).float().to(self.device)
    dataY = torch.from_numpy(dataY).float().to(self.device)

    for n in range(epochs):
      # Train the discriminator.
      # -----------------------------------------------------------------------
      # We want the discriminator to learn the mapping from tests to test
      # outputs.
      D_loss = self.loss(self.modelD(dataX), dataY)
      self.optimizerD.zero_grad()
      self.optimizerD.step()

      # Visualize the computational graph.
      #print(make_dot(D_loss, params=dict(self.modelD.named_parameters())))

      # Train the generator.
      # -----------------------------------------------------------------------
      # We generate noise and label it to have output 1. Training the generator
      # in this way should shift it to generate examples with high output
      # values (high fitness).
      noise_tests = 8 # TODO: make configurable
      noise = ((torch.rand(size=(noise_tests, self.modelG.input_shape)) - 0.5)/0.5).to(self.device)
      outputs = self.modelD(self.modelG(noise))
      fake_label = torch.ones(size=(noise_tests, 1)).to(self.device)

      # Notice the following subtlety. Above the tensor 'outputs' contains
      # information on how it is computed (the computation graph is being kept
      # track off) up to the original input 'noise' which does not anymore
      # depend on previous operations. Since 'self.modelD' is used as part of
      # the computation, its parameters are present in the computation graph.
      # These parameters are however not updated because the optimizer is
      # initialized only for the parameters of 'self.modelG' (see the
      # initialization of 'self.modelG'.

      G_loss = self.loss(outputs, fake_label)
      self.optimizerG.zero_grad()
      G_loss.backward()
      self.optimizerG.step()

      # Visualize the computational graph.
      #print(make_dot(G_loss, params=dict(self.modelG.named_parameters())))

  def generate_test(self, N=1):
    """
    Generate N random tests.

    Args:
      N (int): Number of tests to be generated.

    Returns:
      output (np.ndarray): Array of shape (N, self.sut.ndimensions).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")

    # Generate uniform noise in [-1, 1].
    noise = ((torch.rand(size=(N, self.noise_dim)) - 0.5)/0.5).to(self.device)
    return self.modelG(noise).cpu().detach().numpy()

  def predict_fitness(self, test):
    """
    Predicts the fitness of the given test.

    Args:
      test (np.ndarray): Array of shape (N, self.sut.ndimensions).

    Returns:
      output (np.ndarray): Array of shape (N, 1).
    """

    if len(test.shape) != 2 or test.shape[1] != self.sut.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.sut.ndimensions))

    test_tensor = torch.from_numpy(test).float().to(self.device)
    return self.modelD(test_tensor).cpu().detach().numpy()

class RandomGenerator(Model):
  """
  Implements the random test generator.
  """

  def __init__(self, sut, device):
    super().__init__()

    self.sut = sut
    self.device = device

  def train_with_batch(self, dataX, dataY, epochs=1):
    pass

  def generate_test(self, N=1):
    """
    Generate N random tests.

    Args:
      N (int): Number of tests to be generated.

    Returns:
      output (np.ndarray): Array of shape (N, self.sut.ndimensions).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")

    return np.random.uniform(-1, 1, (N, self.sut.ndimensions))

  def predict_fitness(self, test):
    """
    Predicts the fitness of the given test.

    Args:
      test (np.ndarray): Array of shape (N, self.sut.ndimensions).

    Returns:
      output (np.ndarray): Array of shape (N, 1).
    """

    if len(test.shape) != 2 or test.shape[1] != self.sut.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.sut.ndimensions))

    return np.ones(shape=(test.shape[0], 1))

