#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import numpy as np
from sklearn.model_selection import train_test_split

from config import config, get_model, test_pretty_print
from logger import Logger
from analyzer import *

def experiment(analyzer_seq):
  """
  Run an experiment measuring the performance of all analyzer models described
  in analyzer_seq.
  """

  # This is for easy access to models, loggers, etc. using functions in other
  # modules.
  model_id = "wgan" # ogan, wgan, random
  sut_id = "sbst" # odroid, sbst_validator, sbst

  enable_log_printout = False

  train_settings = {}
  train_settings["nn"] = {"analyzer_epochs": 10}
  train_settings["nnw"] = {"analyzer_epochs": 10}
  train_settings["adaboost"] = {}
  train_settings["randomforest"] = {}
  train_settings["gbt"] = {}
  train_settings["svr"] = {}
  train_settings["knn"] = {}

  # Select training and validation sets randomly.
  random_sampling = True
  # How much is used for training.
  training_tests = 150
  # How much is used for validation.
  validation_tests = 50

  def loss(RY, PY):
    """
    Check for how large a percentage the prediction is off by 0.25 in absolute
    sense.
    """

    A = np.abs(RY-PY)
    return (A >= 0.25).sum()/RY.shape[0]

  logger = Logger(quiet=not enable_log_printout)
  model, _view_test, _save_test = get_model(sut_id, model_id, logger)

  losses = {}

  analyzer_map = {"nn": Analyzer_NN,
                  "nnw": Analyzer_NN_weighted,
                  "adaboost": Analyzer_AdaBoost,
                  "randomforest": Analyzer_RandomForest,
                  "gbt": Analyzer_GradientBoosting,
                  "svr": Analyzer_SVR,
                  "knn": Analyzer_KNN}

  analyzers = [analyzer_map[A](model.sut.ndimensions, model.device, logger) for A in analyzer_seq]
  for n in range(len(analyzer_seq)):
    if analyzer_seq[n] == "nn" or analyzer_seq[n] == "nnw":
      analyzers[n].modelA.train(False)

  logger.log("Loading pregenerated tests...")
  with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
    X = np.load(f)
    Y = np.load(f)

  if random_sampling:
    train_X, validation_X, train_Y, validation_Y = train_test_split(X, Y, train_size=training_tests, test_size=validation_tests)
  else:
    train_X = X[:training_tests]
    train_Y = Y[:training_tests]
    validation_X = X[training_tests:training_tests+validation_tests]
    validation_Y = Y[training_tests:training_tests+validation_tests]

  logger.log("Training the analyzers...")
  for n, A in enumerate(analyzer_seq):
    analyzers[n].train_with_batch(train_X,
                                  train_Y,
                                  train_settings=train_settings[A],
                                  log=True)
    logger.log("Trained {}.".format(A))

  logger.log("\nPerformance on validation data:")
  logger.log("Test: Real:" + "".join(" {}:".format(A) for A in analyzer_seq))
  for n in range(validation_X.shape[0]):
    #if validation_Y[n,0] < 0.5: continue
    test = validation_X[n].reshape(1, validation_X.shape[1])
    predictions = [validation_Y[n,0]] + [analyzers[n].predict(test)[0,0] for n in range(len(analyzers))]
    s = test_pretty_print(test)
    for p in predictions:
      s += " {:.2f}".format(p)
    logger.log(s)

  logger.log("\nLosses:")
  logger.log("-"*80)
  for n, A in enumerate(analyzer_seq):
    l = loss(validation_Y, analyzers[n].predict(validation_X))
    losses[A] = l
    logger.log("{}: {}".format(A, l))

  return losses

if __name__ == "__main__":
  #analyzer_seq = ["nn", "nnw", "adaboost", "randomforest", "svr", "knn"]
  analyzer_seq = ["nn", "knn", "randomforest", "gbt"]
  results = {A:[] for A in analyzer_seq}

  """
  For now we compute the loss of each analyzer algorithm on randomly selected
  training and validation data and report the mean and std over 500
  repetitions.
  """

  for n in range(500):
    print(n)
    losses = experiment(analyzer_seq)
    for A in analyzer_seq:
      results[A].append(losses[A])

  for A in analyzer_seq:
    mean = np.array(results[A]).mean()
    std = np.array(results[A]).std()
    print("{}: {}, {}".format(A, mean, std))
