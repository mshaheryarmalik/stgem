#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, datetime

import imageio

import numpy as np

from config import *
from logger import Logger

if __name__ == "__main__":
  model_id = "ogan"
  sut_id = "odroid" # odroid, sbst_validator, sbst

  enable_log_printout = True
  enable_view = True
  enable_save = True

  # Initialize the system under test and validator.
  # ---------------------------------------------------------------------------
  logger = Logger(quiet=not enable_log_printout)
  model, _view_test, _save_test = get_model(sut_id, model_id, logger)

  session = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
  session_directory = os.path.join(config[sut_id][model_id]["test_save_path"], session)
  view_test = lambda t: _view_test(t) if enable_view else None
  save_test = lambda t, f: _save_test(t, session, f) if enable_save else None
  os.makedirs(os.path.join(config[sut_id][model_id]["test_save_path"], session), exist_ok=True)

  # Generate initial tests randomly.
  # ---------------------------------------------------------------------------
  test_inputs = np.zeros(shape=(model.N_tests, model.sut.ndimensions)) # array to hold all generated tests
  test_outputs = np.zeros(shape=(model.N_tests, 1))                    # array to hold test outputs
  tests_generated = 0                                                  # how many tests are generated so far

  # TODO: put to config
  roads_similarity_tolerance = 0.00

  def detect_similar_roads(test_n):

    for n in range(tests_generated):
      # distance compare option 1
      difference = np.sqrt(np.sum((test_inputs[n,:] - test_n)**2))

      # distance compare option 2
      # difference = np.sqrt(np.sum(np.abs((test_inputs[n,:] - test_n))))

      # distance compare option 3
      # difference = np.max(np.abs(test_inputs[n,:] - test_n))

      if difference <= roads_similarity_tolerance:
        # print('too similar -----------------------------------------------------------', difference)
        return True

    return False

  load = False
  if load:
    logger.log("Loading pregenerated initial tests.")
    with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
      test_inputs[:model.random_init,:] = np.load(f)[:model.random_init,:]
      test_outputs[:model.random_init,:] = np.load(f)[:model.random_init,:]
      tests_generated = model.random_init
  else:
    logger.log("Generating and running {} random valid tests.".format(model.random_init))
    while tests_generated < model.random_init:
      test = model.sut.sample_input_space()
      if model.validity(test)[0,0] == 0: continue
      test_inputs[tests_generated,:] = test
      tests_generated += 1

      view_test(test)
      # TODO: add initial zeros
      save_test(test, "init_{}".format(tests_generated))

      logger.log("Executing {} ({}/{})".format(test, tests_generated, model.random_init))
      test_outputs[tests_generated - 1,:] = model.sut.execute_test(test)
      logger.log("Result: {}".format(test_outputs[tests_generated - 1,0]))

  # Train the model with initial tests.
  # ---------------------------------------------------------------------------
  logger.log("Training model...")
  model.train_with_batch(test_inputs[:tests_generated,:],
                         test_outputs[:tests_generated,:],
                         epoch_settings=model.epoch_settings_init)

  # Begin the main loop for new test generation and training.
  # ---------------------------------------------------------------------------
  while tests_generated < model.N_tests:
    # Generate a new valid test with high fitness and decrease target fitness
    # as per execution of the loop.
    # -------------------------------------------------------------------------
    logger.log("Starting to generate a new test.")
    target_fitness = 1
    rounds = 0
    invalid = 0
    while True:
      new_test = model.generate_test()
      rounds += 1

      # Check if the test is valid.
      if model.validity(new_test)[0,0] == 0:
        invalid += 1
        continue

      if detect_similar_roads(new_test):
        continue

      # Predict the fitness of the new test.
      new_fitness = model.predict_fitness(new_test)[0,0]

      target_fitness *= 0.95 # TODO: make configurable

      # Check if the new test has high enough fitness.
      if new_fitness >= target_fitness: break

    # Add the new test to our test suite.
    # -------------------------------------------------------------------------
    test_inputs[tests_generated,:] = new_test
    tests_generated += 1

    logger.log("Chose test {} with predicted fitness {}. Generated total {} tests of which {} were invalid.".format(new_test, new_fitness, rounds + 1, invalid))
    view_test(new_test)
    save_test(new_test, "test_{}".format(len(test_inputs) + 1))

    # Actually run the new test on the SUT.
    logger.log("Executing the test...")

    test_outputs[tests_generated - 1,:] = model.sut.execute_test(new_test)

    logger.log("The actual fitness {} for the generated test.".format(test_outputs[tests_generated - 1,0]))

    # Train the model with the new test.
    logger.log("Training the model...")
    model.train_with_batch(test_inputs[:tests_generated,:],
                           test_outputs[:tests_generated,:],
                           epoch_settings=model.epoch_settings)

  # Train the model on the complete collected data.
  # ---------------------------------------------------------------------------
  # TODO

  # Evaluate the generated tests.
  # ---------------------------------------------------------------------------
  total = tests_generated
  logger.log("Generated total {} tests.".format(total))

  total_positive = sum(1 for n in range(tests_generated) if test_outputs[n,0] >= model.sut.target)
  logger.log("{}/{} ({} %) are positive.".format(total_positive, total, round(total_positive/total*100, 1)))

  fitness = model.predict_fitness(test_inputs)
  total_predicted_positive = sum(fitness >= target_fitness)[0]
  logger.log("{}/{} ({} %) are predicted to be positive".format(total_predicted_positive, total, round(total_predicted_positive/total*100, 1)))

  # Save the final model, training data, log, etc.
  # ---------------------------------------------------------------------------
  # Save training parameters.
  with open(os.path.join(session_directory, "parameters"), mode="w") as f:
    f.write(json.dumps(model.parameters))

  # Save training data.
  with open(os.path.join(session_directory, "training_data.npy"), mode="wb") as f:
    np.save(f, test_inputs)
    np.save(f, test_outputs)

  # Save the trained model.
  model.save(session_directory)

  # Save the log.
  logger.save(os.path.join(session_directory, "session.log"))

  # Create an animation out of the generated roads.
  if sut_id in ["sbst_validator", "sbst"]:
    session_directory = os.path.join(config[sut_id][model_id]["test_save_path"], session)
    training_images = []
    for filename in os.listdir(session_directory):
      if filename.startswith("init") or filename.startswith("test"):
        training_images.append(imageio.imread(os.path.join(session_directory, filename)))

    imageio.mimsave(os.path.join(session_directory, "training_clip.gif"), training_images)

