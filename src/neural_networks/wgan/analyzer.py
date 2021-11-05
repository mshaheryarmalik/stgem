#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class AnalyzerNetwork(nn.Module):
  """
  Define the neural network model for the WGAN analyzer.
  """

  def __init__(self, input_shape, neurons):
    super(AnalyzerNetwork, self).__init__()

    # The dimension of the input vector.
    self.input_shape = input_shape
    # Number of neurons per layer.
    self.neurons = neurons

    # We use three fully connected layers with self.neurons many neurons.
    self.alayer1 = nn.Linear(self.input_shape, self.neurons)
    self.alayer2 = nn.Linear(self.neurons, self.neurons)
    self.alayer3 = nn.Linear(self.neurons, 1)
    #self.bn1 = nn.BatchNorm1d(self.neurons)
    #self.bn2 = nn.BatchNorm1d(self.neurons)

  def forward(self, x):
    x = F.relu(self.alayer1(x))
    x = F.relu(self.alayer2(x))
    #x = F.relu(self.bn1(self.alayer1(x)))
    #x = F.relu(self.bn2(self.alayer2(x)))
    x = torch.sigmoid(self.alayer3(x))

    return x
