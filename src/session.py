#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, datetime, json

from config import config

class Session:
  """
  A class for accessing and saving session parameters.
  """

  def __init__(self, model_id, sut_id, session_id=None):
    self.model_id = model_id
    self.sut_id = sut_id
    self.id = session_id if session_id is not None else datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    self.session_directory = os.path.join(config[sut_id][model_id]["test_save_path"], self.id)

    self.saved_parameters = ["id"]

    # Ensure that the session directory is created.
    os.makedirs(os.path.join(config[self.sut_id][self.model_id]["test_save_path"], self.id), exist_ok=True)

  @property
  def parameters(self):
    return {k:getattr(self, k) for k in self.saved_parameters if hasattr(self, k)}

  def add_saved_parameter(self, *parameters):
    for parameter in parameters:
      if not parameter in self.saved_parameters:
        self.saved_parameters.append(parameter)

  def remove_saved_parameter(self, *parameters):
    for parameter in parameters:
      try:
        self.saved_parameters.remove(parameter)
      except ValueError:
        pass

  def load(self):
    """
    Load the saved parameters from the appropriate file in the session
    directory.
    """

    file_name = os.path.join(self.session_directory, "session_parameters")
    if not os.path.exists(file_name):
      raise ValueError("File {} not found.".format(file_name))

    with open(file_name, mode="r") as f:
      data = json.load(f)

    for a in self.saved_parameters:
      if not a in data:
        #raise ValueError("Attribute {} not defined in {}.".format(a, file_name))
        setattr(self, a, None)
      else:
        setattr(self, a, data[a])
