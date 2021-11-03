#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os, json

import numpy as np

from config import config, get_model

if __name__ == "__main__":
  if len(sys.argv) < 4:
    print("The command line arguments should specify sut_id, model_id, and a session directory.")
    raise SystemExit

  sut_id = sys.argv[1]
  if not sut_id in config["available_sut"]:
    raise ValueError("The sut_id '{}' is invalid.".format(sut_id))

  model_id = sys.argv[2]
  if not model_id in config["available_model"]:
    raise ValueError("The model_id '{}' is invalid.".format(model_id))

  session_directory = sys.argv[3]
  session = os.path.basename(os.path.normpath(session_directory))

  if not os.path.exists(session_directory):
    print("Directory '{}' does not exist.".format(session_directory))

  model, _view_test, _save_test = get_model(sut_id, model_id)
  model.load("final", session_directory)

  view_test = lambda t: _view_test(t)
  save_test = lambda t, f: _save_test(t, session, f)

  # Generate new samples to assess quality visually.
  total = 1000
  print("Generating {} new tests...".format(total))
  new_tests = model.generate_test(total)
  print("Covariance matrix of the generated tests:")
  print(np.cov(new_tests, rowvar=False))
  for n in range(30):
    #view_test(new_tests[n,:])
    save_test(new_tests[n,:], "eval_{}".format(n + 1))

  fitness = model.predict_fitness(new_tests)
  total_predicted_positive = sum(fitness >= model.sut.target)[0]
  print("{}/{} ({} %) are predicted to be positive".format(total_predicted_positive, total, round(total_predicted_positive/total*100, 1)))

