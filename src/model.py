#!/usr/bin/python3
# -*- coding: utf-8 -*-

class Model:
  """
  Base class for all models.
  """

  def __init__(self, sut, parameters, logger=None):
    self.sut = sut
    self.parameters = parameters
    self.logger = logger
    self.log = lambda s: self.logger.model.info(s) if logger is not None else None

  def __getattr__(self, name):
    value = self.parameters.get(name)
    if value is None:
      raise AttributeError(name)

    return value

  def generate_test(self):
    raise NotImplementedError()

