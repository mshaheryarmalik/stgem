#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, datetime

from config import config, get_model
from logger import Logger
from session import Session
from algorithms import main_ogan, main_wgan, main_random

if __name__ == "__main__":
  model_id = "wgan" # ogan, wgan, random
  sut_id = "sbst" # odroid, sbst_validator, sbst_plane, sbst_dave2, sbst

  # TODO: Put to config.
  random_init = 60
  N_tests = 300

  enable_log_printout = True
  enable_view = False
  enable_save = False
  pretrained_analyzer = False
  model_snapshot = True
  load_pregenerated_data = False

  # Initialize the model and viewing and saving mechanisms.
  # ---------------------------------------------------------------------------
  logger = Logger(quiet=not enable_log_printout)
  model, _view_test, _save_test = get_model(sut_id, model_id, logger)

  if model_id in ["ogan", "wgan"] and random_init == 1:
    print("Error: The training does not work when random_init = 1.")
    raise SystemExit
  if model_id == "wgan" and random_init < model.batch_size:
    print("Warning: The training may work unexpectedly when the number of initial random tests ({}) is smaller than the batch size ({}).".format(random_init, model.batch_size))

  session = Session(model_id, sut_id)
  session.load_pregenerated_data = load_pregenerated_data
  session.random_init = random_init
  session.N_tests = N_tests
  if session.N_tests < session.random_init:
    raise ValueError("The total number of tests should be larger than the number of random initial tests.")
  session.add_saved_parameter(*config["session_attributes"][model_id])
  view_test = lambda t: _view_test(t) if enable_view else None
  save_test = lambda t, f: _save_test(t, session.id, f) if enable_save else None

  # Call the actual training code.
  # ---------------------------------------------------------------------------
  call_dict = {"ogan":main_ogan,
               "wgan":main_wgan,
               "random":main_random}
  test_inputs, test_outputs = call_dict[model_id](model_id, sut_id, model, session, view_test, save_test, pretrained_analyzer, model_snapshot)

  # Evaluate the generated tests.
  # ---------------------------------------------------------------------------
  total = session.N_tests
  logger.log("Generated total {} tests.".format(total))

  total_positive = sum(1 for n in range(total) if test_outputs[n,0] >= model.sut.target)
  logger.log("{}/{} ({} %) are positive.".format(total_positive, total, round(total_positive/total*100, 1)))

  fitness = model.predict_fitness(test_inputs)
  total_predicted_positive = sum(fitness >= model.sut.target)[0]
  logger.log("{}/{} ({} %) are predicted to be positive".format(total_predicted_positive, total, round(total_predicted_positive/total*100, 1)))

