#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch

from sut import OdroidSUT
from models import GAN, RandomGenerator

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Initialize the system under test. Which is Odroid for now.
  output_data = 1
  fitness_threshold = 6.0
  sut = OdroidSUT(output_data, fitness_threshold)
  # Initialize the model.
  model = GAN(sut, device)
  #model = RandomGenerator(sut, device)

  """
  Here we begin to sample new tests from the input space and begin the
  generation of new tests. The SUT methods return numpy arrays, but we convert
  the tests to lists of tuples (which are hashable) and maintain a dictionary
  of used tests for efficiency.
  """

  # Generate initial tests randomly.
  # ---------------------------------------------------------------------------
  random_init = 50 # TODO: put to config
  # TODO: In principle we should check here that no test is obtained twice.
  #       Currently the implementation ensures that this does not happen.
  test_inputs = []
  test_outputs = []
  test_visited = {}

  _dataX, _dataY = sut.execute_random_tests(random_init)
  for n, test in enumerate(_dataX):
    test_inputs.append(tuple(test))
    test_outputs.append(_dataY[n,0])
    test_visited[tuple(test)] = True
  del _dataX, _dataY

  # Train the model with initial tests.
  # ---------------------------------------------------------------------------
  # TODO: epochs? Now it's 10 without any specific reason.
  model.train_with_batch(np.array(test_inputs), np.array(test_outputs).reshape(len(test_outputs), 1), epochs=10)

  # Begin the main loop for new test generation and training.
  # ---------------------------------------------------------------------------
  # How many tests are generated.
  N = 200 # TODO: put to config/parameter
  while len(test_inputs) < N:
    # Generate a new test with high fitness and decrease target fitness as per
    # execution of the loop.
    target_fitness = 1
    rounds = 0
    while True:
      # Generate a new test (from noise), but do not use a test that has
      # already been used.
      new_test = model.generate_test()
      # TODO: in order to traverse the test space more completely, we probably
      #       should exclude tests that are "too close" to tests already
      #       generated. Ivan's code does this.
      if tuple(new_test[0,:]) in test_visited: continue

      # Predict the fitness of the new test.
      new_fitness = model.predict_fitness(new_test)[0,0]

      target_fitness *= 0.95 # TODO: make configurable

      # Check if the new test has high enough fitness.
      if new_fitness >= target_fitness: break
      rounds += 1

    # Add the new test to our test suite.
    test_inputs.append(tuple(new_test[0,:]))
    test_visited[tuple(new_test[0,:])] = True
    # Actually run the new test on the SUT.
    test_outputs.append(model.sut.execute_test(new_test)[0,0])
    #print(test_inputs[-1], rounds)

    # Train the model with the new test.
    # Set use_final = 1 to train with the new test only, not with the whole
    # test suite generated so far.
    model.train_with_batch(np.array(test_inputs), np.array(test_outputs).reshape(len(test_outputs), 1), discriminator_epochs=100, use_final=-1)

  # Evaluate the generated tests.
  # ---------------------------------------------------------------------------
  total = len(test_inputs)
  print("Generated total {} tests.".format(total))

  total_positive = sum(1 for output in test_outputs if output >= sut.target)
  print("{}/{} ({} %) are positive.".format(total_positive, total, round(total_positive/total*100, 1)))

  fitness = model.predict_fitness(np.array(test_inputs))
  total_predicted_positive = sum(fitness >= target_fitness)[0]
  print("{}/{} ({} %) are predicted to be positive".format(total_predicted_positive, total, round(total_predicted_positive/total*100, 1)))
