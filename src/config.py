#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

config = {}
config["data_path"] = os.path.join("..", "data")
config["test_save_path"] = os.path.join("..", "simulations")

for model in ["ogan", "wgan"]:
  # Config common to all systems under test.
  for sut_id in ["odroid_{}".format(model), "sbst_validator_{}".format(model), "sbst_{}".format(model)]:
    config[sut_id] = {}
    config[sut_id]["data_directory"] = os.path.join(config["data_path"], sut_id)
    config[sut_id]["pregenerated_initial_data"] = os.path.join(config[sut_id]["data_directory"], "pregenerated_initial_data.npy")

  # SUT specific configs.
  config["odroid_{}".format(model)] = {"file_base": os.path.join(config["data_path"], "odroid", "odroid")}

  config["sbst_{}".format(model)]["beamng_home"] = "C:\\Users\\japel\\dev\\BeamNG"

def convert(test):
  """
  Convenience function for converting an array of shape (1, N) or (N) to an
  array of shape (N).
  """

  if test.shape[0] == 1:
    return test.reshape(test.shape[1])
  else:
    if len(test.shape) == 1:
      return test
    else:
      raise ValueError("The input must have shape (1, N) or (N).")

