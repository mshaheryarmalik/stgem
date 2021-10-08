#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class ValidatorNetwork(nn.Module):
  """
  Define the neural network model for a validator.
  """

  def __init__(self, input_shape, neurons):
    super(ValidatorNetwork, self).__init__()

    # The dimension of the input vector.
    self.input_shape = input_shape
    # Number of neurons per layer.
    self.neurons = neurons

    # We use three fully connected layers with self.neurons many neurons.
    self.dlayer1 = nn.Linear(self.input_shape, self.neurons)
    self.dlayer2 = nn.Linear(self.neurons, self.neurons)
    self.dlayer3 = nn.Linear(self.neurons, 1)

  def forward(self, x):
    x = F.relu(self.dlayer1(x))
    x = F.relu(self.dlayer2(x))
    x = torch.sigmoid(self.dlayer3(x))

    return x

