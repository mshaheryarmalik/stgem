#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, sys

import numpy as np

from config import config, get_model

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

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

  file_name = sys.argv[3]
  if os.path.exists(file_name):
    print("File {} already exists.".format(file_name))
    raise SystemExit

  model, _view_test, _save_test = get_model(sut_id, model_id)

  view_test = lambda t: _view_test(t)
  save_test = lambda t, f: _save_test(t, session, f)

  # Load the pregenerated data.
  with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
    data_X = np.load(f)
    data_Y = np.load(f)

  # Put the data into bins based on their fitness.
  bins = 10
  get_bin = lambda x: int(x*bins) if x < 1.0 else bins-1
  test_bins = {n:[] for n in range(bins)}
  for n in range(len(data_X)):
    test_bins[get_bin(data_Y[n,0])].append(n)

  # Find bin frequencies for balanced sampling.
  frequencies = [len(test_bins[n])/len(data_X) for n in range(bins)]

  # Sample a balanced training data.
  N = len(test_bins[np.argmin(frequencies)]) * bins
  sample_X = np.zeros(shape=(N, data_X.shape[1]))
  sample_Y = np.zeros(shape=(N, data_Y.shape[1]))
  for n, bin in enumerate(np.random.choice(list(range(bins)), N, p=frequencies)):
    i = np.random.choice(test_bins[bin])
    sample_X[n] = data_X[i]
    sample_Y[n] = data_Y[i]

  X, X_validation, Y, Y_validation = train_test_split(sample_X, sample_Y, test_size=0.2)
  X_validation = torch.from_numpy(X_validation).float().to(model.device)
  Y_validation = torch.from_numpy(Y_validation).float().to(model.device)
  del data_X
  del data_Y

  dataloader = DataLoader(TensorDataset(torch.Tensor(X), torch.Tensor(Y)), batch_size=32, shuffle=True)
  del X
  del Y

  class EarlyStop:

    def __init__(self, patience, delta):
      self.patience = patience
      self.delta = delta
      self.best_loss = None
      self.counter = 0

    def __call__(self, losses_training, losses_validation):
      loss = losses_validation[-1]
      if self.best_loss is None:
        self.best_loss = loss
      elif self.best_loss - loss > self.delta:
        self.best_loss = loss
        self.counter = 0
      else:
        self.counter += 1

      return self.counter >= self.patience

  stop = EarlyStop(patience=10, delta=0)

  model.analyzer.analyzer_learning_rate = 0.00001
  model.analyzer.neurons = 32
  model.analyzer.initialize()

  train_settings = {"analyzer_epochs": 5}

  epochs = 10
  stopped = False
  for epoch in range(epochs):
    losses_training = []
    losses_validation = []
    rounds = 0
    for X, Y in dataloader:
      loss_training = model.analyzer.train_with_batch(X.cpu().detach().numpy(), Y.cpu().detach().numpy(), train_settings=train_settings)
      loss_validation = model.analyzer.analyzer_loss(X_validation, Y_validation)

      print("Epoch {}/{}: Iteration {}: Training loss: {}, Validation loss: {}".format(epoch + 1, epochs, rounds + 1, loss_training, loss_validation))

      losses_training.append(loss_training)
      losses_validation.append(loss_validation)

      if stop(losses_training, losses_validation):
        print("Early stop.")
        stopped = True
        break

      rounds += 1

    if stopped: break

    rounds += 1

  print("Best loss: {}".format(stop.best_loss.item()))

  # Save the analyzer.
  torch.save(model.analyzer.modelA.state_dict(), file_name)
