#!/usr/bin/env python
# -*- coding: utf-8 -*-

class SUT:
  """
  Base class implementing a system under test.
  """

  def __init__(self):
    self.ndimensions = None
    self.dataX = None
    self.dataY = None

  def execute_test(self, tests):
    raise NotImplementedError()

  def execute_random_test(self, N=1):
    raise NotImplementedError()

  def sample_input_space(self, N=1):
    raise NotImplementedError()

