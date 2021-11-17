#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, sys

import numpy as np

from config import config

if __name__ == "__main__":
  if len(sys.argv) < 5:
    print("The command line arguments should specify sut_id, model_id, and a session directory.")
    raise SystemExit

  sut_id = sys.argv[1]
  if not sut_id in config["available_sut"]:
    raise ValueError("The sut_id '{}' is invalid.".format(sut_id))

  model_id = sys.argv[2]
  if not model_id in config["available_model"]:
    raise ValueError("The model_id '{}' is invalid.".format(model_id))

  new_training_data_file = sys.argv[3]
  if not os.path.exists(new_training_data_file):
    raise ValueError("Directory '{}' does not exist.".format(new_training_data_file))

  try:
    N = int(sys.argv[4])
  except ValueError:
    raise ValueError("A positive integer specifying how many initial tests to be used must be specified.")
  if N <= 0:
    raise ValueError("A positive integer specifying how many initial tests to be used must be specified.")

  new_file_name = "new.npy"
  if os.path.exists(new_file_name):
    print("The file '{}' already exists.".format(new_file_name))
    raise SystemExit

  with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="rb") as f:
    X_old = np.load(f)
    Y_old = np.load(f)

  with open(new_training_data_file, mode="rb") as f:
    X_new = np.load(f)
    Y_new = np.load(f)

  X = np.zeros(shape=(X_old.shape[0] + N, X_old.shape[1]))
  Y = np.zeros(shape=(X.shape[0], Y_old.shape[1]))

  X[:X_old.shape[0]] = X_old
  Y[:Y_old.shape[0]] = Y_old

  X[X_old.shape[0]:] = X_new[:N]
  Y[Y_old.shape[0]:] = Y_new[:N]

  with open(new_file_name, mode="wb") as f:
    np.save(f, X)
    np.save(f, Y)

  print("Wrote the new pregenerated data to '{}'.".format(new_file_name))

