#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os, json
from math import log10

import numpy as np

import torch

from config import config, get_model
from session import Session

def evaluate_session(session):
  print("Reporting on session {}".format(session.id))
  print()

  # Load the session training data.
  with open(os.path.join(session.session_directory, "training_data.npy"), mode="rb") as f:
    X = np.load(f)
    Y = np.load(f)

  # Report positive tests etc.
  report(X, Y)

  # Report analyzer performance over the experiment.
  test_analyzer()
  #analyzer_performance()
  return

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

def report(X, Y):
  total = Y.shape[0]
  total_positive = sum(Y >= model.sut.target)[0]
  print("{}/{} ({} %) tests are positive.".format(total_positive, total, round(total_positive/total*100, 1)))
  total_noninitial_positive = sum(Y[session.random_init:,] >= model.sut.target)[0]
  print("{}/{} ({} %) non-initial tests are positive.".format(total_noninitial_positive, total, round(total_noninitial_positive/total*100, 1)))
  avg = np.mean(Y)
  print("The test suite has average fitness {}.".format(avg))
  avg_initial = np.mean(Y[:session.random_init])
  print("The initial tests have average fitness {}.".format(avg_initial))
  avg_noninitial = np.mean(Y[session.random_init:])
  print("The noninitial tests have average fitness {}.".format(avg_noninitial))
  window = 10
  print("Moving averages with window size {}.".format(window))
  mavg = []
  for n in range(window - 1, session.N_tests - 1):
    mavg.append(round(np.mean(Y[n - window + 1:n + 1]), 2))
  print(mavg)

def test_analyzer():
  N = 50
  with open(os.path.join(session.session_directory, "training_data.npy"), mode="rb") as f:
    data_X = np.load(f)
    data_Y = np.load(f)

  with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
    X_valid = torch.from_numpy(np.load(f)).float().to(model.device)
    Y_valid = torch.from_numpy(np.load(f)).float().to(model.device)

  K = 20
  X_valid = X_valid[:K]
  Y_valid = Y_valid[:K]

  model.initialize()

  train_settings = {"analyzer_epochs": 5}
  for n in range(session.random_init, session.N_tests):
    X = data_X[:n]
    Y = data_Y[:n]
    model.train_analyzer_with_batch(X, Y, train_settings=train_settings)
    set_X = torch.from_numpy(X).float().to(model.device)
    set_Y = torch.from_numpy(Y).float().to(model.device)
    weights = model.analyzer.weights(set_Y)
    loss1 = model.analyzer.analyzer_loss(set_X, set_Y, weights)
    weights = model.analyzer.weights(Y_valid)
    loss2 = model.analyzer.analyzer_loss(X_valid, Y_valid, weights)
    print("{}: loss on training: {}, loss on validation: {}".format(n, loss1, loss2))

  set_X = torch.from_numpy(data_X).float().to(model.device)
  set_Y = torch.from_numpy(data_Y).float().to(model.device)
  weights = model.analyzer.weights(set_Y)
  loss1 = model.analyzer.analyzer_loss(set_X, set_Y, weights)
  weights = model.analyzer.weights(Y_valid)
  loss2 = model.analyzer.analyzer_loss(X_valid, Y_valid, weights)
  print("Final loss on training: {}, final loss on validation: {}".format(loss1, loss2))

  print()
  print(Y_valid.reshape(-1))
  out = model.analyzer.modelA(X_valid)
  print(out.reshape(-1))

def analyzer_performance():
  # Get N random samples from pregenerated data.
  N = 50
  #with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
  with open(os.path.join(session.session_directory, "training_data.npy"), mode="rb") as f:
    data_X = np.load(f)
    data_Y = np.load(f)
  """
  idx = np.random.choice(data_X.shape[0], N)
  X_valid = torch.from_numpy(data_X[idx, :]).float().to(model.device)
  Y_valid = torch.from_numpy(data_Y[idx, :]).float().to(model.device)
  """
  X_valid = torch.from_numpy(data_X[:N]).float().to(model.device)
  Y_valid = torch.from_numpy(data_Y[:N]).float().to(model.device)
  del data_X
  del data_Y

  zeros = lambda s, N: (s + "{{:0{}d}}").format(int(log10(session.N_tests)) + 1).format(N)

  threshold = 0.7
  for n in range(session.random_init + 1, session.N_tests):
    model.load(zeros("model_snapshot_", n), session.session_directory)
    idx1 = (Y_valid < threshold).reshape(-1)
    set_Y = Y_valid[idx1,:]
    set_X = X_valid[idx1,:]
    weights = model.analyzer.weights(set_Y)
    loss1 = model.analyzer.analyzer_loss(set_X, set_Y, weights)

    idx2 = (Y_valid >= threshold).reshape(-1)
    set_Y = Y_valid[idx2,:]
    set_X = X_valid[idx2,:]
    weights = model.analyzer.weights(set_Y)
    loss2 = model.analyzer.analyzer_loss(set_X, set_Y, weights)
    print("{}: {}, {}; {} {}".format(n, loss1, loss2, sum(idx1), sum(idx2)))

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

  sessions = []
  if sys.argv[3] == "-d":
    if len(sys.argv) < 5:
      raise ValueError("A directory must be specified after -d.")
    for directory in os.listdir(sys.argv[4]):
      if not directory.startswith("2021"): continue
      session = Session(model_id, sut_id, directory)
      session.add_saved_parameter(*config["session_attributes"][model_id])
      session.session_directory = os.path.join(sys.argv[4], directory)
      session.load()
      sessions.append(session)
  else:
    session_id = os.path.basename(os.path.normpath(sys.argv[3]))
    session = Session(model_id, sut_id, session_id)
    session.add_saved_parameter(*config["session_attributes"][model_id])
    session.session_directory = sys.argv[3]
    session.load()
    sessions.append(session)

  model, _view_test, _save_test = get_model(sut_id, model_id)

  view_test = lambda t: _view_test(t)
  save_test = lambda t, f: _save_test(t, session, f)

  for session in sessions:
    evaluate_session(session)
