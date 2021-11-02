#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, datetime

from config import config, get_model
from logger import Logger
from algorithms import main_ogan, main_wgan, main_random

if __name__ == "__main__":
  model_id = "random" # ogan, wgan, random
  sut_id = "odroid" # odroid, sbst_validator, sbst

  enable_log_printout = True
  enable_view = True
  enable_save = True
  load_pregenerated_data = False

  # Initialize the model and viewing and saving mechanisms.
  # ---------------------------------------------------------------------------
  logger = Logger(quiet=not enable_log_printout)
  model, _view_test, _save_test = get_model(sut_id, model_id, logger)

  session = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
  session_directory = os.path.join(config[sut_id][model_id]["test_save_path"], session)
  view_test = lambda t: _view_test(t) if enable_view else None
  save_test = lambda t, f: _save_test(t, session, f) if enable_save else None
  os.makedirs(os.path.join(config[sut_id][model_id]["test_save_path"], session), exist_ok=True)

  # Call the actual training code.
  # ---------------------------------------------------------------------------
  call_dict = {"ogan":main_ogan,
               "wgan":main_wgan,
               "random":main_random}
  test_inputs, test_outputs = call_dict[model_id](model_id, sut_id, model, session, session_directory, view_test, save_test, load_pregenerated_data)

  # Evaluate the generated tests.
  # ---------------------------------------------------------------------------
  total = model.N_tests
  logger.log("Generated total {} tests.".format(total))

  total_positive = sum(1 for n in range(total) if test_outputs[n,0] >= model.sut.target)
  logger.log("{}/{} ({} %) are positive.".format(total_positive, total, round(total_positive/total*100, 1)))

  fitness = model.predict_fitness(test_inputs)
  total_predicted_positive = sum(fitness >= model.sut.target)[0]
  logger.log("{}/{} ({} %) are predicted to be positive".format(total_predicted_positive, total, round(total_predicted_positive/total*100, 1)))

