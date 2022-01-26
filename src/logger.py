#!/usr/bin/python3
# -*- coding: utf-8 -*-

class Logger:

  def __init__(self, quiet=False):
    self.quiet = quiet

    self.total_log = ""

  def log(self, msg, quiet=None):
    self.total_log += msg + "\n"
    if (quiet is not None and not quiet) or not self.quiet:
      print(msg)

  def save(self, file_name):
    with open(file_name, mode="w") as f:
      f.write(self.total_log)

