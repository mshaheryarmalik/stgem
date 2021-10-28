#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, datetime, json

import numpy as np

from config import config, get_model
from logger import Logger

if __name__ == "__main__":
  model_id = "wgan"
  sut_id = "odroid" # odroid, sbst_validator, sbst

  enable_log_printout = True
  enable_view = True
  enable_save = True

  # Initialize the model and viewing and saving mechanisms.
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
  test_critic_training = []                                            # list which indicates which rows of test_inputs belong to the critic training data

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

      # TODO: add mechanism for selecting test which is in some sense different
      #       from previous tests

      view_test(test)
      # TODO: add initial zeros
      save_test(test, "init_{}".format(tests_generated))

      logger.log("Executing {} ({}/{})".format(test, tests_generated, model.random_init))
      test_outputs[tests_generated - 1,:] = model.sut.execute_test(test)
      logger.log("Result: {}".format(test_outputs[tests_generated - 1,0]))

  init_threshold = 0.1 # TODO: make configurable
  # Use the tests whose fitness exceeds init_threshold as training data for the
  # critic.
  # TODO: What if the number of samples in the training data is very low? What
  #       if it's zero?
  test_critic_training = [n for n in range(tests_generated) if test_outputs[n,0] >= init_threshold]
  # TODO: We are now exiting in case we have nothing to traing with. Should we
  #       redo the initial phase if the number of samples is too low?
  if len(test_critic_training) == 0:
    logger.log("No training samples found for the critic.")
    raise SystemExit

  # TODO: Place this somewhere else.
  def report_critic():
    global test_inputs
    global test_outputs
    global tests_generated
    global test_critic_training

    data = test_outputs[test_critic_training,:]
    mu = data.mean(0)[0]
    sigma = data.std(0)[0]
    logger.log("Critic training data has {} samples with mean {} and std {}.".format(len(test_critic_training), mu, sigma))

  report_critic()

  # Train the model with initial tests.
  # ---------------------------------------------------------------------------
  logger.log("Training model...")
  model.train_with_batch(test_inputs[:tests_generated,:],
                         test_outputs[:tests_generated,:],
                         test_inputs[test_critic_training,:],
                         test_outputs[test_critic_training,:],
                         epoch_settings=model.epoch_settings_init,
                         log=False)

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

      # TODO: add mechanism for selecting test which is in some sense different
      #       from previous tests

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
    save_test(new_test, "test_{}".format(tests_generated))

    # Actually run the new test on the SUT.
    logger.log("Executing the test...")

    test_outputs[tests_generated - 1,:] = model.sut.execute_test(new_test)

    logger.log("The actual fitness {} for the generated test.".format(test_outputs[tests_generated - 1,0]))

    # Update critic training data.
    # -------------------------------------------------------------------------
    # Now we simply add the new test if it is above the mean minus one standard
    # deviation.
    data = test_outputs[test_critic_training,:]
    mu = data.mean(0)[0]
    sigma = data.std(0)[0]
    o = test_outputs[tests_generated - 1,0]
    if o >= mu or mu - o <= sigma:
      logger.log("Added the new test to the critic training data.")
      test_critic_training.append(tests_generated - 1)
      
      # Consider if we should remove the test with the lowest fitness from the
      # training data.
      if np.random.random() <= 0.25:
        i = np.argmin(data)
        o = test_outputs[test_critic_training[i],0]
        if np.abs(o - mu) >= 0.1:
          logger.log("Removing test {} with fitness {}.".format(test_inputs[test_critic_training[i],:], o))
          test_critic_training.pop(i)
    else:
      logger.log("Didn't add the new test to the critic training data.")

    report_critic()

    # Train the model.
    # -------------------------------------------------------------------------
    logger.log("Training the model...")
    model.train_with_batch(test_inputs[:tests_generated,:],
                         test_outputs[:tests_generated,:],
                         test_inputs[test_critic_training,:],
                         test_outputs[test_critic_training,:],
                         epoch_settings=model.epoch_settings,
                         log=True)

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
    np.save(f, np.array(test_critic_training))

  # Save the trained model.
  model.save(session_directory)

  # Save the log.
  logger.save(os.path.join(session_directory, "session.log"))

