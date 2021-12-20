#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from matplotlib import pyplot as plt

import torch

config = {}
config["available_sut"] = ["odroid", "sbst_validator", "sbst_plane", "sbst_dave2", "sbst"]
config["available_model"] = ["ogan", "wgan", "random"]
config["data_path"] = os.path.join("..", "data")
config["test_save_path"] = os.path.join("..", "simulations")

# Config for all SUT's.
# ----------------------------------------------------------------------------
for sut_id in config["available_sut"]:
  config[sut_id] = {}
  # Common config to all models.
  config[sut_id]["data_directory"] = os.path.join(config["data_path"], sut_id)
  # Model-specific config.
  for model_id in config["available_model"]:
    config[sut_id][model_id] = {}
    config[sut_id][model_id]["algorithm_version"] = 1
    config[sut_id][model_id]["pregenerated_initial_data"] = os.path.join(config[sut_id]["data_directory"], "pregenerated_initial_data_{}.npy".format(model_id))
    config[sut_id][model_id]["test_save_path"] = os.path.join(config["test_save_path"], "{}_{}".format(sut_id, model_id))
    config[sut_id][model_id]["train_settings_init"] = {}
    config[sut_id][model_id]["train_settings"] = {}
    config[sut_id][model_id]["train_settings_post"] = {}

  # OGAN defaults.
  config[sut_id]["ogan"]["noise_dim"] = 100
  config[sut_id]["ogan"]["gan_neurons"] = 128
  config[sut_id]["ogan"]["gan_learning_rate"] = 0.001
  config[sut_id]["ogan"]["analyzer_learning_rate"] = 0.001
  config[sut_id]["ogan"]["analyzer_neurons"] = 32
  config[sut_id]["ogan"]["train_settings_init"] = {"epochs": 2,
                                                   "discriminator_epochs": 20,
                                                   "generator_epochs": 1}
  config[sut_id]["ogan"]["train_settings"] = {"epochs": 1,
                                              "discriminator_epochs": 5,
                                              "generator_epochs": 1}

  # WGAN defaults.
  # 1 = something, 2 = increasing removal probability, 3 = WGAN weighted sampling, 4 = buckets
  config[sut_id]["wgan"]["algorithm_version"] = 4
  config[sut_id]["wgan"]["noise_dim"] = 10
  config[sut_id]["wgan"]["gan_neurons"] = 128
  config[sut_id]["wgan"]["gan_learning_rate"] = 0.00005
  config[sut_id]["wgan"]["analyzer_learning_rate"] = 0.001
  config[sut_id]["wgan"]["analyzer_neurons"] = 32
  config[sut_id]["wgan"]["gp_coefficient"] = 10
  config[sut_id]["wgan"]["batch_size"] = 32
  config[sut_id]["wgan"]["train_settings_init"] = {"epochs": 3,
                                                   "analyzer_epochs": 20,
                                                   "critic_epochs": 5,
                                                   "generator_epochs": 1}
  config[sut_id]["wgan"]["train_settings"] = {"epochs": 2,
                                              "analyzer_epochs": 10,
                                              "critic_epochs": 5,
                                              "generator_epochs": 1}

# SUT-specific configs.
# ----------------------------------------------------------------------------
config["odroid"]["file_base"] = os.path.join(config["odroid"]["data_directory"], "odroid")
config["odroid"]["output"] = 1
config["odroid"]["fitness_threshold"] = 6.0

for sut_id in ["sbst_validator", "sbst_plane", "sbst_dave2", "sbst"]:
  config[sut_id]["beamng_home"] = "C:\\BeamNG\\BeamNG.tech.v0.24.0.1"
  #config[sut_id]["beamng_home"] = "C:\\Users\\japeltom\\BeamNG\\BeamNG.tech.v0.24.0.1"
  #config[sut_id]["beamng_home"] = "C:\\Users\\japel\\dev\\BeamNG"
  config[sut_id]["map_size"] = 200

for sut_id in ["sbst_plane", "sbst_dave2", "sbst"]:
  config[sut_id]["fitness_threshold"] = 0.95
  config[sut_id]["max_speed"] = 300

config["sbst_validator"]["curvature_points"] = 9
config["sbst_plane"]["curvature_points"] = 10
config["sbst_dave2"]["curvature_points"] = 5
config["sbst"]["curvature_points"] = 5

# Session configs.
# ----------------------------------------------------------------------------
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
                                        "N_candidate_tests",
                                        "train_delay",
                                        "bins",
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

    validator_bb = Validator(input_size=config[sut_id]["curvature_points"], validator_bb=lambda t: sbst_validate_test(t, sut))
    sut = SBSTSUT_validator(map_size=config[sut_id]["map_size"], curvature_points=validator_bb.ndimensions, validator_bb=validator_bb)
    validator = None

    def _view_test(test, sut):
      fig = sbst_test_to_image(convert(test), sut)
      fig.show()
      plt.close(fig)

    def _save_test(test, session, file_name, sut):
      fig = sbst_test_to_image(convert(test), sut)
      fig.savefig(os.path.join(config[sut_id][model_id]["test_save_path"], session, file_name + ".jpg"))
      plt.close(fig)

  elif sut_id == "sbst_plane":
    from sut.sut_sbst import SBSTSUT_plane, sbst_test_to_image, sbst_validate_test
    from validator.validator import Validator

    sut = SBSTSUT_plane(beamng_home=config[sut_id]["beamng_home"],
                        map_size=config[sut_id]["map_size"],
                        curvature_points=config[sut_id]["curvature_points"],
                        oob_tolerance=config[sut_id]["fitness_threshold"],
                        max_speed=config[sut_id]["max_speed"])
    validator = Validator(sut.ndimensions, lambda t: sbst_validate_test(t, sut))

    def _view_test(test, sut):
      fig = sbst_test_to_image(convert(test), sut)
      fig.show()
      plt.close(fig)

    def _save_test(test, session, file_name, sut):
      fig = sbst_test_to_image(convert(test), sut)
      fig.savefig(os.path.join(config[sut_id][model_id]["test_save_path"], session, file_name + ".jpg"))
      plt.close(fig)

  elif sut_id == "sbst_dave2":
    from sut.sut_sbst import SBSTSUT_dave2, sbst_test_to_image, sbst_validate_test
    from validator.validator import Validator

    sut = SBSTSUT_dave2(beamng_home=config[sut_id]["beamng_home"],
                        map_size=config[sut_id]["map_size"],
                        curvature_points=config[sut_id]["curvature_points"],
                        oob_tolerance=config[sut_id]["fitness_threshold"],
                        max_speed=config[sut_id]["max_speed"])
    validator = Validator(sut.ndimensions, lambda t: sbst_validate_test(t, sut))

    def _view_test(test, sut):
      fig = sbst_test_to_image(convert(test), sut)
      fig.show()
      plt.close(fig)

    def _save_test(test, session, file_name, sut):
      fig = sbst_test_to_image(convert(test), sut)
      fig.savefig(os.path.join(config[sut_id][model_id]["test_save_path"], session, file_name + ".jpg"))
      plt.close(fig)

  elif sut_id == "sbst":
    from sut.sut_sbst import SBSTSUT_beamng, sbst_test_to_image, sbst_validate_test
    from validator.validator import Validator

    sut = SBSTSUT_beamng(beamng_home=config[sut_id]["beamng_home"],
                         map_size=config[sut_id]["map_size"],
                         curvature_points=config[sut_id]["curvature_points"],
                         oob_tolerance=config[sut_id]["fitness_threshold"],
                         max_speed=config[sut_id]["max_speed"])
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

  if model_id in ["ogan", "wgan"]:
    model.noise_dim = config[sut_id][model_id]["noise_dim"]
    model.gan_neurons = config[sut_id][model_id]["gan_neurons"]
    model.gan_learning_rate = config[sut_id][model_id]["gan_learning_rate"]
    model.analyzer_learning_rate = config[sut_id][model_id]["analyzer_learning_rate"]
    model.analyzer_neurons = config[sut_id][model_id]["analyzer_neurons"]

  if model_id == "wgan":
    model.gp_coefficient = config[sut_id][model_id]["gp_coefficient"]
    model.batch_size = config[sut_id][model_id]["batch_size"]

  model.initialize()

  return model, lambda t: _view_test(t, sut), lambda t, s, f: _save_test(t, s, f, sut)
