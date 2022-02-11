#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorNetwork(nn.Module):
  """
  Define the neural network model for the GAN generator.
  """

  def __init__(self, noise_dim, output_shape, neurons):
    super(GeneratorNetwork, self).__init__()

    # The dimension of the input vector.
    self.input_shape = noise_dim
    # The dimension of the output vector.
    self.output_shape = output_shape
    # Number of neurons per layer.
    self.neurons = neurons

    # We use three fully connected layers with self.neurons many neurons.
    self.glayer1 = nn.Linear(self.input_shape, self.neurons)
    self.glayer2 = nn.Linear(self.neurons, self.neurons)
    self.glayer3 = nn.Linear(self.neurons, self.output_shape)
    # Use uniform Glorot initialization of weights as in Keras.
    torch.nn.init.xavier_uniform_(self.glayer1.weight)
    torch.nn.init.xavier_uniform_(self.glayer2.weight)
    torch.nn.init.xavier_uniform_(self.glayer3.weight)

  def forward(self, x):
    x = F.relu(self.glayer1(x))
    x = F.relu(self.glayer2(x))
    x = torch.tanh(self.glayer3(x)) # Squash the output values to [-1, 1].

    return x

class DiscriminatorNetwork(nn.Module):
  """
  Define the neural network model for the GAN discriminator.
  """

  def __init__(self, input_shape, neurons, discriminator_output_activation="linear"):
    super(DiscriminatorNetwork, self).__init__()

    # The dimension of the input vector.
    self.input_shape = input_shape
    # Number of neurons per layer.
    self.neurons = neurons

    # We use three fully connected layers with self.neurons many neurons.
    self.dlayer1 = nn.Linear(self.input_shape, self.neurons)
    self.dlayer2 = nn.Linear(self.neurons, self.neurons)
    self.dlayer3 = nn.Linear(self.neurons, 1)
    # Use uniform Glorot initialization of weights as in Keras.
    torch.nn.init.xavier_uniform_(self.dlayer1.weight)
    torch.nn.init.xavier_uniform_(self.dlayer2.weight)
    torch.nn.init.xavier_uniform_(self.dlayer3.weight)

    # Select the output activation function.
    a = discriminator_output_activation
    if a == "linear":
        self.output_activation = torch.nn.Identity()
    elif a == "sigmoid":
        self.output_activation = torch.sigmoid
    else:
        raise Exception("Unknown output activation function '{}'.".format(a))

  def forward(self, x):
    x = F.leaky_relu(self.dlayer1(x), negative_slope=0.1) # LeakyReLU recommended in the literature for GANs discriminators.
    x = F.leaky_relu(self.dlayer2(x), negative_slope=0.1)
    x = self.output_activation(self.dlayer3(x))

    return x

