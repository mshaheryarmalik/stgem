#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch

config = {}
config["data_path"] = os.path.join("..", "data")
config["test_save_path"] = os.path.join("..", "simulations")

for sut_id in ["odroid", "sbst_validator", "sbst"]:
  config[sut_id] = {}
  # Common config to all models.
  config[sut_id]["data_directory"] = os.path.join(config["data_path"], sut_id)
  # Model-specific config.
  for model_id in ["ogan", "wgan"]:
    config[sut_id][model_id] = {}
    config[sut_id][model_id]["pregenerated_initial_data"] = os.path.join(config[sut_id]["data_directory"], "pregenerated_initial_data_{}.npy".format(model_id))
    config[sut_id][model_id]["test_save_path"] = os.path.join(config["test_save_path"], "{}_{}".format(sut_id, model_id))

  config[sut_id]["ogan"]["epoch_settings_init"] = {"epochs": 2,
                                                   "discriminator_epochs": 20,
                                                   "generator_epochs": 1}
  config[sut_id]["ogan"]["epoch_settings"] = {"epochs": 1,
                                              "discriminator_epochs": 5,
                                              "generator_epochs": 1}

  config[sut_id]["wgan"]["epoch_settings_init"] = {"epochs": 2,
                                                   "analyzer_epochs": 20,
                                                   "critic_epochs": 5,
                                                   "generator_epochs": 1}
  config[sut_id]["wgan"]["epoch_settings"] = {"epochs": 1,
                                              "analyzer_epochs": 5,
                                              "critic_epochs": 5,
                                              "generator_epochs": 1}

# SUT-specific configs.
config["odroid"]["file_base"] = os.path.join(config["odroid"]["data_directory"], "odroid")
config["odroid"]["output"] = 1
config["odroid"]["fitness_threshold"] = 6.0
config["odroid"]["random_init"] = 50
config["odroid"]["N_tests"] = 200

config["sbst"]["beamng_home"] = "C:\\Users\\japel\\dev\\BeamNG"
config["sbst"]["map_size"] = 200
config["sbst"]["curvature_points"] = 5
config["sbst"]["random_init"] = 50
config["sbst"]["N_tests"] = 200

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

def get_model(sut_id, model_id, logger=None):
  """
  Return a complete initialized model based on SUT id and model id.
  """

  if sut_id == "odroid":
    from sut.sut_odroid import OdroidSUT

    sut = OdroidSUT(config["odroid"]["output"], config["odroid"]["fitness_threshold"])
    validator = None

    random_init = config["odroid"]["random_init"]
    N_tests = config["odroid"]["N_tests"]

    def _view_test(test, sut):
      pass

    def _save_test(test, session, file_name, sut):
      pass

  elif sut_id == "sbst_validator":
    from sut.sut_sbst import SBSTSUT_beamng, SBSTSUT_validator, sbst_test_to_image, sbst_validate_test
    from validator.validator import Validator

    validator_bb = Validator(input_size=config["sbst"]["curvature_points"], validator_bb=lambda t: sbst_validate_test(t, sut))
    sut = SBSTSUT_validator(map_size=config["sbst"]["map_size"], curvature_points=validator_bb.ndimensions, validator_bb=validator_bb)
    validator = None

    random_init = config["sbst"]["random_init"]
    N_tests = config["sbst"]["N_tests"]
    if N_tests < random_init:
      raise ValueError("The total number of tests should be larger than the number of random initial tests.")

    def _view_test(test, sut):
      plt = sbst_test_to_image(convert(test), sut)
      plt.show()

    def _save_test(test, session, file_name, sut):
      plt = sbst_test_to_image(convert(test), sut)
      plt.savefig(os.path.join(config[sut_id][model_id]["test_save_path"], session, file_name + ".jpg"))

  elif sut_id == "sbst":
    from sut.sut_sbst import SBSTSUT_beamng, SBSTSUT_validator, sbst_test_to_image, sbst_validate_test
    from validator.validator import Validator

    sut = SBSTSUT_beamng(config["sbst"]["beamng_home"], map_size=config["sbst"]["map_size"], curvature_points=config["sbst"]["curvature_points"])
    validator = Validator(sut.ndimensions, lambda t: sbst_validate_test(t, sut))

    random_init = config["sbst"]["random_init"]
    N_tests = config["sbst"]["N_tests"]

    def _view_test(test, sut):
      plt = sbst_test_to_image(convert(test), sut)
      plt.show()

    def _save_test(test, session, file_name, sut):
      plt = sbst_test_to_image(convert(test), sut)
      plt.savefig(os.path.join(config[sut_id][model_id]["test_save_path"], session, file_name + ".jpg"))

  else:
    raise Exception("Unknown sut id '{}'.".format(sut_id))

  if model_id == "ogan":
    from models import OGAN
    C = OGAN
  elif model_id == "wgan":
    from models import WGAN
    C = WGAN
  elif model_id == "random":
    from models import RandomGenerator
    C = RandomGenerator
  else:
    raise Exception("Unknown model id '{}'.".format(model_id))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = C(sut, validator, device, logger)

  # Set training parameters.
  model.random_init = random_init
  model.N_tests = N_tests
  model.epoch_settings_init = config[sut_id][model_id]["epoch_settings_init"]
  model.epoch_settings = config[sut_id][model_id]["epoch_settings"]

  return model, lambda t: _view_test(t, sut), lambda t, s, f: _save_test(t, s, f, sut)

