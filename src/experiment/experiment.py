#!/usr/bin/python3
# -*- coding: utf-8 -*-

class Experiment:
  """
  Base class for experiments.
  """

  def __init__(self, algorithm):
    self.algorithm = algorithm

  def start(self):
    raise NotImplementedError()

