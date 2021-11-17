#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from matplotlib import pyplot as plt

import torch

config = {}
config["available_sut"] = ["odroid", "sbst_validator", "sbst"]
config["available_model"] = ["ogan", "wgan", "random"]
config["data_path"] = os.path.join("..", "data")
config["test_save_path"] = os.path.join("..", "simulations")

for sut_id in config["available_sut"]:
  config[sut_id] = {}
  # Common config to all models.
  config[sut_id]["data_directory"] = os.path.join(config["data_path"], sut_id)
  config[sut_id]["gp_coefficient"] = 10
  # Model-specific config.
  for model_id in config["available_model"]:
    config[sut_id][model_id] = {}
    config[sut_id][model_id]["algorithm_version"] = 1
    config[sut_id][model_id]["pregenerated_initial_data"] = os.path.join(config[sut_id]["data_directory"], "pregenerated_initial_data_{}.npy".format(model_id))
    config[sut_id][model_id]["test_save_path"] = os.path.join(config["test_save_path"], "{}_{}".format(sut_id, model_id))
    config[sut_id][model_id]["train_settings_init"] = {}
    config[sut_id][model_id]["train_settings"] = {}
    config[sut_id][model_id]["train_settings_post"] = {}

  config[sut_id]["ogan"]["train_settings_init"] = {"epochs": 2,
                                                   "discriminator_epochs": 20,
                                                   "generator_epochs": 1}
  config[sut_id]["ogan"]["train_settings"] = {"epochs": 1,
                                              "discriminator_epochs": 5,
                                              "generator_epochs": 1}
  config[sut_id]["ogan"]["train_settings_post"] = {"epochs": 2,
                                                   "discriminator_epochs": 20,
                                                   "generator_epochs": 1}

  config[sut_id]["wgan"]["algorithm_version"] = 2
  config[sut_id]["wgan"]["train_settings_init"] = {"epochs": 2,
                                                   "analyzer_epochs": 20,
                                                   "critic_epochs": 5,
                                                   "generator_epochs": 1}
  config[sut_id]["wgan"]["train_settings"] = {"epochs": 1,
                                              "analyzer_epochs": 5,
                                              "critic_epochs": 5,
                                              "generator_epochs": 1}
  config[sut_id]["wgan"]["train_settings_post"] = {"epochs": 10,
                                                   "analyzer_epochs": 0,
                                                   "critic_epochs": 5,
                                                   "generator_epochs": 1}

# SUT-specific configs.
config["sbst_validator"]["wgan"]["train_settings_init"]["epochs"] = 50
config["sbst_validator"]["wgan"]["train_settings_init"]["analyzer_epochs"] = 20
config["sbst_validator"]["wgan"]["train_settings_init"]["critic_epochs"] = 10
config["sbst_validator"]["wgan"]["train_settings_init"]["generator_epochs"] = 1
config["sbst_validator"]["wgan"]["train_settings"]["analyzer_epochs"] = 10

config["sbst"]["wgan"]["train_settings_init"]["epochs"] = 3
config["sbst"]["wgan"]["train_settings_init"]["analyzer_epochs"] = 20
config["sbst"]["wgan"]["train_settings_init"]["critic_epochs"] = 10
config["sbst"]["wgan"]["train_settings_init"]["generator_epochs"] = 1
config["sbst"]["wgan"]["train_settings"]["analyzer_epochs"] = 10

config["odroid"]["file_base"] = os.path.join(config["odroid"]["data_directory"], "odroid")
config["odroid"]["output"] = 1
config["odroid"]["fitness_threshold"] = 6.0

config["sbst"]["beamng_home"] = "C:\\BeamNG\\BeamNG.research.v1.7.0.1"
#config["sbst"]["beamng_home"] = "C:\\Users\\japel\\dev\\BeamNG"
config["sbst"]["map_size"] = 200
config["sbst"]["curvature_points"] = 5

config["session_attributes"] = {}
config["session_attributes"]["ogan"] = ["N_tests",
                                        "random_init",
                                        "fitness_coef",
                                        "load_pregenerated_data",
                                        "N_tests_generated",
                                        "N_invalid_tests_generated",
                                        "N_positive_tests",
                                        "fitness_avg",
                                        "fitness_std",
                                        "time_total",
                                        "time_training_total",
                                        "time_execution_total",
                                        "time_training",
                                        "time_generation",
                                        "time_execution"]
config["session_attributes"]["wgan"] = ["N_tests",
                                        "random_init",
                                        "fitness_coef",
                                        "init_fitness_threshold",
                                        "post_fitness_threshold",
                                        "N_candidate_tests",
                                        "removal_probability_1",
                                        "removal_probability_2",
                                        "load_pregenerated_data",
                                        "critic_training_data_history",
                                        "N_tests_generated",
                                        "N_invalid_tests_generated",
                                        "N_positive_tests",
                                        "fitness_avg",
                                        "fitness_std",
                                        "time_total",
                                        "time_training_total",
                                        "time_execution_total",
                                        "time_training",
                                        "time_generation",
                                        "time_execution"]
config["session_attributes"]["random"] = ["N_tests",
                                          "random_init",
                                          "N_tests_generated",
                                          "N_invalid_tests_generated",
                                          "N_positive_tests",
                                          "fitness_avg",
                                          "fitness_std",
                                          "time_total",
                                          "time_training_total",
                                          "time_execution_total",
                                          "time_training",
                                          "time_generation",
                                          "time_execution"]

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

def test_pretty_print(test):
  """
  Returns a fixed-length string representing the given test.
  """

  test = convert(test)
  s = "["
  for n in range(test.shape[0]):
    s += "{: .2f}, ".format(test[n])

  return s[:-2] + "]"

def get_model(sut_id, model_id, logger=None):
  """
  Return a complete initialized model based on SUT id and model id.
  """

  if sut_id == "odroid":
    from sut.sut_odroid import OdroidSUT

    sut = OdroidSUT(config["odroid"]["output"], config["odroid"]["fitness_threshold"])
    validator = None

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

    def _view_test(test, sut):
      fig = sbst_test_to_image(convert(test), sut)
      fig.show()
      plt.close(fig)

    def _save_test(test, session, file_name, sut):
      fig = sbst_test_to_image(convert(test), sut)
      fig.savefig(os.path.join(config[sut_id][model_id]["test_save_path"], session, file_name + ".jpg"))
      plt.close(fig)

  elif sut_id == "sbst":
    from sut.sut_sbst import SBSTSUT_beamng, SBSTSUT_validator, sbst_test_to_image, sbst_validate_test
    from validator.validator import Validator

    sut = SBSTSUT_beamng(config["sbst"]["beamng_home"], map_size=config["sbst"]["map_size"], curvature_points=config["sbst"]["curvature_points"])
    validator = Validator(sut.ndimensions, lambda t: sbst_validate_test(t, sut))

    def _view_test(test, sut):
      fig = sbst_test_to_image(convert(test), sut)
      fig.show()
      plt.close(fig)

    def _save_test(test, session, file_name, sut):
      fig = sbst_test_to_image(convert(test), sut)
      fig.savefig(os.path.join(config[sut_id][model_id]["test_save_path"], session, file_name + ".jpg"))
      plt.close(fig)

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
  model.algorithm_version = config[sut_id][model_id]["algorithm_version"]
  model.train_settings_init = config[sut_id][model_id]["train_settings_init"]
  model.train_settings = config[sut_id][model_id]["train_settings"]
  model.train_settings_post = config[sut_id][model_id]["train_settings_post"]

  if model_id == "wgan":
    model.gp_coefficient = config[sut_id]["gp_coefficient"]

  return model, lambda t: _view_test(t, sut), lambda t, s, f: _save_test(t, s, f, sut)
