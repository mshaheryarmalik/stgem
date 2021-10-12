#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, os

import numpy as np
import torch

from config import *
from sut.sut_sbst import SBSTSUT_beamng, sbst_test_to_image, sbst_validate_test
from validator.validator import Validator

from code_pipeline.validation import TestValidator
from code_pipeline.tests_generation import RoadTestFactory

def sample_and_visualize(N, sut, validator):
  """
  Sample and visualize tests.
  """

  for n in range(N):
    for test in sut.sample_input_space():
      print("Test {}".format(n))
      print("  sampled test: {}".format(test))
      valid = validator.validate(test.reshape(1, test.shape[0]))
      print("  test valid: {}".format(valid[0][0]))
      print("  test as points: {}".format(sut.test_to_road_points(test)))
      print()

      test_to_image(test)

def generate_validator_training_data(N, sut, validator, append=True):
  """
  Generate data to train a validator.
  """

  data = []
  if append and os.path.exists(config["sbst"]["validator_training_data"]):
    data = np.load(config["sbst"]["validator_training_data"]).tolist()

  for n in range(N):
    for test in sut.sample_input_space():
        v = validator.validity(test.reshape(1, test.shape[0]))
        row = list(test) + [int(v[0])]
        if not row in data:
          data.append(row)

  np.save(config["sbst"]["validator_training_data"], data)

def train_validator(validator, save=False):
  """
  Train the neural network for the validator using a precollected training
  data.
  """

  data = np.load(config["sbst"]["validator_training_data"])
  dataX = data[:,0:-1]
  dataY = data[:,-1].reshape(data.shape[0], 1)
  validator.train_with_batch(dataX, dataY, epochs=100)
  if save:
    torch.save(validator.modelV.state_dict(), config["sbst"]["validator_neural_network"])

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  sut = SBSTSUT_beamng(config["sbst"]["beamng_home"], map_size=200, curvature_points=5)
  validator = Validator(sut, lambda t: sbst_validate_test(t, sut), device=device)
  test_to_image = lambda t: sbst_test_to_image(test, sut)

  #sample_and_visualize(10, sut, validator)
  #generate_validator_training_data(2000, sut, validator, append=True)

  # Train/load a neural network for the validator.
  train = False
  if train:
    train_validator(validator, save=True)
  else:
    validator.load(config["sbst"]["validator_neural_network"])

  raise SystemExit

  # Test the validator model on new data.
  t = time.time()
  N = 20
  correct = 0
  for n in range(N):
    for test in sut.sample_input_space():
      test = test.reshape(1, test.shape[0])
      real = validator.validity(test)
      predicted = validator.predict_validity(test)
      p = 0 if predicted[0][0] < 0.5 else 1
      if real[0][0] == p: correct += 1
      print(test, real, predicted)
      test = test.reshape((test.shape[-1]))
      test_to_image(test)
      #if predicted >= 0.5:
      #  sut.execute_test(test)

  print(correct/N)
  raise SystemExit

  # Execute a single test.
  test = np.array([[3.20000000e+01, -1.50326268e-02, 1.93185060e-02, 5.29393106e-02, 1.51600207e-03]]).reshape((5))
  test_to_image(test.reshape((5)))
  result = sut.execute_test(test.reshape(1, 5))
  print("Test result: {}".format(result))
  raise SystemExit

