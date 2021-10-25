#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import torch

from config import *
from validator.validator import Validator
from models import WGAN

def log(msg):
  print(msg)

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # odroid_wgan, sbst_validator_wgan, sbst_wgan
  sut_id = "sbst_validator_wgan"

  enable_log = True
  enable_view = True
  enable_save = True

  # Initialize the system under test and validator.
  # ---------------------------------------------------------------------------
  # We assume that there exists an efficient and perfect validator oracle. This
  # oracle is used mainly for test validation, and the trained proxy for it is
  # only used for training.
  if sut_id == "odroid_wgan":
    from sut.sut_odroid import OdroidSUT

    random_init = 50
    N = 200

    output_data = 1
    fitness_threshold = 6.0

    sut = OdroidSUT(output_data, fitness_threshold)
    validator = None

    epoch_settings_init = {"epochs": 2,
                           "analyzer_epochs": 20,
                           "critic_epochs": 5,
                           "generator_epochs": 1}
    epoch_settings = {"epochs": 1,
                      "analyzer_epochs": 5,
                      "critic_epochs": 5,
                      "generator_epochs": 1}

    def _view_test(test):
      pass

    def _save_test(test, file_name):
      pass

  elif sut_id == "sbst_validator_wgan":
    from sut.sut_sbst import SBSTSUT_beamng, SBSTSUT_validator, sbst_test_to_image, sbst_validate_test

    random_init = 50
    N = 200

    map_size = 200
    curvature_points = 5

    validator_bb = Validator(input_size=curvature_points, validator_bb=lambda t: sbst_validate_test(t, sut))
    sut = SBSTSUT_validator(map_size=map_size, curvature_points=validator_bb.ndimensions, validator_bb=validator_bb)
    test_to_image = lambda t, file_name=None: _test_to_image(sut, t.reshape(sut.ndimensions), file_name)
    validator = None

    epoch_settings_init = {"epochs": 2,
                           "analyzer_epochs": 20,
                           "critic_epochs": 5,
                           "generator_epochs": 1}
    epoch_settings = {"epochs": 1,
                      "analyzer_epochs": 10,
                      "critic_epochs": 5,
                      "generator_epochs": 1}

    def _view_test(test):
      plt = sbst_test_to_image(convert(test), sut)
      plt.show()

    def _save_test(test, file_name):
      plt = sbst_test_to_image(convert(test), sut)
      plt.savefig(os.path.join(config["test_save_path"], sut_id, file_name + ".jpg"))

  elif sut_id == "sbst_wgan":
    from sut.sut_sbst import SBSTSUT_beamng, SBSTSUT_validator, sbst_test_to_image, sbst_validate_test

    random_init = 50
    N = 200

    map_size = 200
    curvature_points = 5

    sut = SBSTSUT_beamng(config["sbst"]["beamng_home"], map_size=map_size, curvature_points=curvature_points)
    validator = Validator(sut.ndimensions, lambda t: sbst_validate_test(t, sut), device=device)

    def _view_test(test):
      plt = sbst_test_to_image(convert(test), sut)
      plt.show()

    def _save_test(test, file_name):
      plt = sbst_test_to_image(convert(test), sut)
      plt.savefig(os.path.join(config["test_save_path"], sut_id, file_name + ".jpg"))

  else:
    print("No SUT specified.")
    raise SystemExit

  view_test = lambda t: _view_test(convert(t)) if enable_view else None
  save_test = lambda t, f: _save_test(convert(t), f) if enable_save else None

  # Initialize the model.
  # ---------------------------------------------------------------------------
  model = WGAN(sut, validator, device)

  """
  Here we begin to sample new tests from the input space and begin the
  generation of new tests. The SUT methods return numpy arrays, but we convert
  the tests to lists of tuples (which are hashable) and maintain a dictionary
  of used tests for efficiency.
  """

  # Generate initial tests randomly.
  # ---------------------------------------------------------------------------
  # TODO: figure out an efficient way to handle the data
  test_inputs = []
  test_outputs = []
  test_critic_training = []

  load = False
  if load:
    log("Loading pregenerated initial tests.")
    data = np.load(config[sut_id]["pregenerated_initial_data"]).tolist()
    for test in data:
      test_inputs.append(tuple(test[:-1]))
      test_outputs.append(test[-1])
    del data
  else:
    log("Generating and running {} random valid tests.".format(random_init))
    while len(test_inputs) < random_init:
      test = sut.sample_input_space()
      if model.validity(test)[0,0] == 0: continue
      test_inputs.append(tuple(test[0,:]))

      # TODO: add mechanism for selecting test which is in some sense different
      #       from previous tests

      view_test(test)
      save_test(test, "init_{}".format(len(test_inputs)))

      log("Executing {} ({}/{})".format(test, len(test_inputs), random_init))
      test_outputs.append(sut.execute_test(test)[0,0])
      log("Result: {}".format(test_outputs[-1]))

  init_threshold = 0.1 # TODO: make configurable
  # Use the tests whose fitness exceeds init_threshold as training data for the
  # critic.
  test_critic_training = [n for n, o in enumerate(test_outputs) if o >= init_threshold]
  # TODO: What if the number of samples in the training data is very low? What
  #       if it's zero?

  # TODO: Place this somewhere else.
  def report_critic():
    global test_inputs
    global test_outputs
    global test_critic_training

    data = np.array(test_outputs).reshape(len(test_outputs), 1)[test_critic_training,:]
    mu = data.mean(0)[0]
    sigma = data.std(0)[0]
    log("Critic training data has {} samples with mean {} and std {}.".format(len(test_critic_training), mu, sigma))

  report_critic()

  # Train the model with initial tests.
  # ---------------------------------------------------------------------------
  log("Training model...")
  inputs_A = np.array(test_inputs)
  outputs_A = np.array(test_outputs).reshape(len(test_outputs), 1)
  model.train_with_batch(inputs_A,
                         outputs_A,
                         inputs_A[test_critic_training,:],
                         outputs_A[test_critic_training,:],
                         epoch_settings=epoch_settings_init)
  del inputs_A
  del outputs_A

  # Begin the main loop for new test generation and training.
  # ---------------------------------------------------------------------------
  # How many tests are generated.
  while len(test_inputs) < N:
    # Generate a new valid test with high fitness and decrease target fitness
    # as per execution of the loop.
    # -------------------------------------------------------------------------
    #log("Starting to generate a new test.")
    target_fitness = 1
    rounds = 0
    invalid = 0
    while True:
      new_test = model.generate_test()

      # Check if the test is valid.
      if model.validity(new_test)[0,0] == 0:
        invalid += 1
        continue

      # TODO: add mechanism for selecting test which is in some sense different
      #       from previous tests

      # Predict the fitness of the new test.
      new_fitness = model.predict_fitness(new_test)[0,0]

      target_fitness *= 0.95 # TODO: make configurable

      # Check if the new test has high enough fitness.
      if new_fitness >= target_fitness: break
      rounds += 1

    log("Chose test {} with predicted fitness {}. Generated total {} tests of which {} were invalid.".format(new_test, new_fitness, rounds + 1, invalid))
    view_test(new_test)
    save_test(new_test, "test_{}".format(len(test_inputs) + 1))

    # Add the new test to our test suite.
    # -------------------------------------------------------------------------
    test_inputs.append(tuple(new_test[0,:]))
    # Actually run the new test on the SUT.
    #log("Executing the test...")
    test_outputs.append(model.sut.execute_test(new_test)[0,0])
    log("The actual fitness {} for the generated test.".format(test_outputs[-1]))

    # Update critic training data.
    # -------------------------------------------------------------------------
    # Now we simply add the new test if it is below one standard deviation of
    # the mean.
    data = np.array(test_outputs).reshape(len(test_outputs), 1)[test_critic_training,:]
    mu = data.mean(0)[0]
    sigma = data.std(0)[0]
    if new_fitness >= mu or mu - new_fitness <= sigma:
      log("Added the new test to the critic training data.")
      test_critic_training.append(len(test_inputs) - 1)
      
      # Consider if we should remove the test with the lowest fitness from the
      # training data.
      if np.random.random() <= 0.25:
        i = np.argmin(data)
        o = test_outputs[test_critic_training[i]]
        if np.abs(o - mu) >= 0.1:
          log("Removing test {} with fitness {}.".format(test_inputs[test_critic_training[i]], test_outputs[test_critic_training[i]]))
          test_critic_training.pop(i)
    else:
      log("Didn't add the new test to the critic training data.")

    report_critic()

    # Train the model.
    # -------------------------------------------------------------------------
    log("Training the model...")
    inputs_A = np.array(test_inputs)
    outputs_A = np.array(test_outputs).reshape(len(test_outputs), 1)
    model.train_with_batch(inputs_A,
                           outputs_A,
                           inputs_A[test_critic_training,:],
                           outputs_A[test_critic_training,:],
                           epoch_settings=epoch_settings)
    del inputs_A
    del outputs_A

  # Evaluate the generated tests.
  # ---------------------------------------------------------------------------
  total = len(test_inputs)
  log("Generated total {} tests.".format(total))

  total_positive = sum(1 for output in test_outputs if output >= sut.target)
  log("{}/{} ({} %) are positive.".format(total_positive, total, round(total_positive/total*100, 1)))

  fitness = model.predict_fitness(np.array(test_inputs))
  total_predicted_positive = sum(fitness >= target_fitness)[0]
  log("{}/{} ({} %) are predicted to be positive".format(total_predicted_positive, total, round(total_predicted_positive/total*100, 1)))

  # Generate new samples to assess quality visually.
  for n, test in enumerate(model.generate_test(30)):
    view_test(test)
    save_test(test, "eval_{}".format(n + 1))
