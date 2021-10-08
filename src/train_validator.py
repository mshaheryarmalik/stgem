#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, os

import numpy as np
import torch

from sut.sut_sbst import SBSTSUT_beamng
from validator.validator import Validator

from code_pipeline.validation import TestValidator
from code_pipeline.tests_generation import RoadTestFactory
from code_pipeline.visualization import RoadTestVisualizer

def test_to_image(test):
  """
  Visualizes the road (described as points in the plane).
  """

  V = RoadTestVisualizer(map_size=20)
  V.visualize_road_test(RoadTestFactory.create_road_test(test))

def sbst_validate_test(test, sut):
  """
  Tests if the road (described as points in the plane) is valid.
  """

  V = TestValidator(map_size=200)
  the_test = RoadTestFactory.create_road_test(sut.test_to_road_points(test))
  valid, msg = V.validate_test(the_test)
  #print(msg)
  return 1 if valid else 0

if __name__ == "__main__":
  beamng_home = "C:\\Users\\japel\\dev\\BeamNG"
  sut = SBSTSUT_beamng(beamng_home, map_size=200, curvature_points=5)
  validator = Validator(sut, lambda t: sbst_validate_test(t, sut), device="cpu")

  # Sample and visualize tests.
  for n in range(10):
    for test in sut.sample_input_space():
      print("Test {}".format(n))
      print("  sampled test: {}".format(test))
      valid = validator.validate(test.reshape(1, test.shape[0]))
      print("  test valid: {}".format(valid[0][0]))
      test = sut.test_to_road_points(test)
      print("  test as points: {}".format(test))
      print()

      test_to_image(test)

  raise SystemExit

  # Execute a single test.
  test = np.array([[3.20000000e+01, -1.50326268e-02, 1.93185060e-02, 5.29393106e-02, 1.51600207e-03]])
  test_to_image(sut.test_to_road_points(test[0,:]))
  result = sut.execute_test(test)
  print("Test result: {}".format(result))
  raise SystemExit

  # Generate a large amount of training data by random sampling.
  if not os.path.exists("datafile.npy"):
    data = []
  else:
    data = np.load("datafile.npy").tolist()

  for n in range(200):
    for test in sut.sample_input_space(1):
      try:
        v = validator.validate(test.reshape(1, test.shape[0]))
        data.append(list(test) + [int(v[0])])
      except:
        continue

  np.save("datafile.npy", data)
  raise SystemExit

  # Train/load a neural network for the validator.
  file_name = "validator_nn"
  train = False
  if train:
    validator.train(200)
    torch.save(validator.modelV.state_dict(), file_name)
  else:
    validator.modelV.load_state_dict(torch.load(file_name))
    validator.modelV.eval()

  # Test the validator model.
  t = time.time()
  for n in range(20):
    for test in sut.sample_input_space():
      test = test.reshape(1, test.shape[0])
      real = validator.validate(test)
      predicted = validator.predict_validity(test)
      print(test, real, predicted)
      test_to_image(test)
      #if predicted >= 0.5:
      #  sut.execute_test(test)

