#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time

class Algorithm:
  """
  Base class for all test suite generation algorithms.
  """

  def __init__(self, sut, test_repository, objective_func, objective_selector, logger=None):
    self.sut = sut
    self.test_repository = test_repository
    self.objective_func = objective_func
    self.objective_selector = objective_selector

    self.logger = logger
    self.log = lambda s: self.logger.algorithm.info(s) if logger is not None else None

    self.test_suite = []
    self.timers = {}
    self.histories = {}

  def __getattr__(self, name):
    value = self.parameters.get(name)
    if value is None:
      raise AttributeError(name)

    return value

  def get_history(self, id):
    if not id in self.timers:
      raise Exception("No history for the identifier '{}'.".format(id))
    return self.histories[id]

  def save_history(self, id, value, single=False):
    if not single:
      if not id in self.timers:
        self.histories[id] = []
      self.histories[id].append(value)
    else:
      self.histories[id] = value

  def timer_start(self, id):
    # TODO: Implement a good time for all platforms.
    if id in self.timers and self.timers[id] is not None:
      raise Exception("Restarting timer '{}' without resetting.".format(id))

    self.timers[id] = time.monotonic()

  def timer_reset(self, id):
    if not id in self.timers:
      raise Exception("No timer '{}' to be reset.".format(id))
    if self.timers[id] is None:
      raise Exception("Timer '{}' already reset.".format(id))

    time_elapsed = time.monotonic() - self.timers[id]
    self.timers[id] = None

    return time_elapsed

  def timers_hold(self):
    for id, t in self.timers.items():
      if t is not None:
        self.timers[id] = time.monotonic() - self.timers[id]

  def timers_resume(self):
    self.timers_hold()

  def generate_test(self):
    raise NotImplementedError()

