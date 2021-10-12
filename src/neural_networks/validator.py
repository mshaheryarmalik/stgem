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
    self.vlayer1 = nn.Linear(self.input_shape, self.neurons)
    self.vlayer2 = nn.Linear(self.neurons, self.neurons)
    self.vlayer3 = nn.Linear(self.neurons, 1)
    self.bn1 = nn.BatchNorm1d(self.input_shape)
    self.bn2 = nn.BatchNorm1d(self.neurons)
    self.bn3 = nn.BatchNorm1d(self.neurons)

  def forward(self, x):
    x = self.bn1(x)
    x = F.relu(self.bn2(self.vlayer1(x)))
    x = F.relu(self.bn3(self.vlayer2(x)))
    x = torch.sigmoid(self.vlayer3(x))

    return x

