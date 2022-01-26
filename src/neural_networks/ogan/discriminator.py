#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorNetwork(nn.Module):
  """
  Define the neural network model for the GAN discriminator.
  """

  def __init__(self, input_shape, neurons):
    super(DiscriminatorNetwork, self).__init__()

    # The dimension of the input vector.
    self.input_shape = input_shape
    # Number of neurons per layer.
    self.neurons = neurons

    # We use three fully connected layers with self.neurons many neurons.
    self.dlayer1 = nn.Linear(self.input_shape, self.neurons)
    self.dlayer2 = nn.Linear(self.neurons, self.neurons)
    self.dlayer3 = nn.Linear(self.neurons, 1)

  def forward(self, x):
    x = F.leaky_relu(self.dlayer1(x), negative_slope=0.1) # LeakyReLU recommended in the literature for GANs discriminators.
    x = F.leaky_relu(self.dlayer2(x), negative_slope=0.1)
    x = torch.sigmoid(self.dlayer3(x))

    return x

