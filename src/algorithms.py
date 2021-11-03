#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, json
from math import log10

import numpy as np

from config import config

def main_ogan(model_id, sut_id, model, session, session_directory, view_test, save_test, load_pregenerated_data):
  """
  The OGAN algorithm for generating a test suite.
  """

  # Format (s + "{}").format(N) with enough initial zeros.
  zeros = lambda s, N: (s + "{{:0{}d}}").format(int(log10(model.N_tests)) + 1).format(N)

  # TODO: make configurable
  # How much to decrease the target fitness per each round when selecting a
  # new generated test.
  fitness_coef = 0.95

  # Generate initial tests randomly.
  # ---------------------------------------------------------------------------
  test_inputs = np.zeros(shape=(model.N_tests, model.sut.ndimensions)) # array to hold all generated tests
  test_outputs = np.zeros(shape=(model.N_tests, 1))                    # array to hold test outputs
  tests_generated = 0                                                  # how many tests are generated so far

  if load_pregenerated_data:
    model.log("Loading pregenerated initial tests.")
    with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
      test_inputs[:model.random_init,:] = np.load(f)[:model.random_init,:]
      test_outputs[:model.random_init,:] = np.load(f)[:model.random_init,:]
      tests_generated = model.random_init
  else:
    model.log("Generating and running {} random valid tests.".format(model.random_init))
    while tests_generated < model.random_init:
      test = model.sut.sample_input_space()
      if model.validity(test)[0,0] == 0: continue
      test_inputs[tests_generated,:] = test
      tests_generated += 1

      # TODO: add mechanism for selecting test which is in some sense different
      #       from previous tests

      view_test(test)
      save_test(test, zeros("init_", tests_generated))

      model.log("Executing {} ({}/{})".format(test, tests_generated, model.random_init))
      test_outputs[tests_generated - 1,:] = model.sut.execute_test(test)
      model.log("Result: {}".format(test_outputs[tests_generated - 1,0]))

  # Train the model with initial tests.
  # ---------------------------------------------------------------------------
  model.log("Training model...")
  model.train_with_batch(test_inputs[:tests_generated,:],
                         test_outputs[:tests_generated,:],
                         epoch_settings=model.epoch_settings_init,
                         log=True)

  """
  # View and save N generated tests based solely on the training on initial
  # data.
  N = 30
  new_tests = model.generate_test(N)
  for n in range(N):
    save_test(new_tests[n,:], zeros("eval_", n + 1))
  """

  # Begin the main loop for new test generation and training.
  # ---------------------------------------------------------------------------
  while tests_generated < model.N_tests:
    # Generate a new valid test with high fitness and decrease target fitness
    # as per execution of the loop.
    # -------------------------------------------------------------------------
    model.log("Starting to generate a new test.")
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

      # Predict the fitness of the new test.
      new_fitness = model.predict_fitness(new_test)[0,0]

      target_fitness *= fitness_coef

      # Check if the new test has high enough fitness.
      if new_fitness >= target_fitness: break

      # View and save tests that failed to be accepted.
      #view_test(new_test)
      #save_test(new_test, zeros(zeros("failed_", tests_generated), rounds))

    # Add the new test to our test suite.
    # -------------------------------------------------------------------------
    test_inputs[tests_generated,:] = new_test
    tests_generated += 1

    model.log("Chose test {} with predicted fitness {}. Generated total {} tests of which {} were invalid.".format(new_test, new_fitness, rounds + 1, invalid))
    view_test(new_test)
    save_test(new_test, zeros("test_", tests_generated))

    # Actually run the new test on the SUT.
    model.log("Executing the test...")

    test_outputs[tests_generated - 1,:] = model.sut.execute_test(new_test)

    model.log("The actual fitness {} for the generated test.".format(test_outputs[tests_generated - 1,0]))

    # Train the model.
    # -------------------------------------------------------------------------
    model.log("Training the model...")
    model.train_with_batch(test_inputs[:tests_generated,:],
                           test_outputs[:tests_generated,:],
                           epoch_settings=model.epoch_settings,
                           log=True)

  # Save the trained models.
  # ---------------------------------------------------------------------------
  model.save("init", session_directory)

  # Save training data.
  # ---------------------------------------------------------------------------
  with open(os.path.join(session_directory, "training_data.npy"), mode="wb") as f:
    np.save(f, test_inputs)
    np.save(f, test_outputs)

  # Train the model on the complete collected data.
  # ---------------------------------------------------------------------------
  # TODO

  # Save the final model, parameters, log, etc.
  # ---------------------------------------------------------------------------
  # Save the final model.
  #model.save(session_directory)

  # Save training parameters.
  with open(os.path.join(session_directory, "parameters"), mode="w") as f:
    f.write(json.dumps(model.parameters))

  # Save the log.
  model.logger.save(os.path.join(session_directory, "session.log"))

  return test_inputs, test_outputs

def main_wgan(model_id, sut_id, model, session, session_directory, view_test, save_test, load_pregenerated_data):
  """
  The WGAN algorithm for generating a test suite.
  """

  # Format (s + "{}").format(N) with enough initial zeros.
  zeros = lambda s, N: (s + "{{:0{}d}}").format(int(log10(model.N_tests)) + 1).format(N)

  # TODO: make configurable
  # Include tests whose fitness exceed the threshould to the initial critic
  # training data.
  init_fitness_threshold = 0.1
  # Include tests whose fitness exceed the threshould to the final critic
  # training data.
  post_fitness_threshold = 0.5
  # How much to decrease the target fitness per each round when selecting a
  # new generated test.
  fitness_coef = 0.95
  # Probability to remove a sample from the critic training data.
  removal_probability = 0.4
  # Do not remove critic training samples which deviate at most the following
  # from the mean fitness of critic training samples.
  removal_distance = 0.1

  def report_critic(model, test_inputs, test_outputs, test_critic_training):
    """
    Report some statistics on the critic training data.
    """

    data = test_outputs[test_critic_training,:]
    mu = data.mean(0)[0]
    sigma = data.std(0)[0]
    model.log("Critic training data has {} samples with mean {} and std {}.".format(len(test_critic_training), mu, sigma))

  # Generate initial tests randomly.
  # ---------------------------------------------------------------------------
  test_inputs = np.zeros(shape=(model.N_tests, model.sut.ndimensions)) # array to hold all generated tests
  test_outputs = np.zeros(shape=(model.N_tests, 1))                    # array to hold test outputs
  tests_generated = 0                                                  # how many tests are generated so far
  test_critic_training = []                                            # list which indicates which rows of test_inputs belong to the critic training data

  if load_pregenerated_data:
    model.log("Loading pregenerated initial tests.")
    with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
      test_inputs[:model.random_init,:] = np.load(f)[:model.random_init,:]
      test_outputs[:model.random_init,:] = np.load(f)[:model.random_init,:]
      tests_generated = model.random_init
  else:
    model.log("Generating and running {} random valid tests.".format(model.random_init))
    while tests_generated < model.random_init:
      test = model.sut.sample_input_space()
      if model.validity(test)[0,0] == 0: continue
      test_inputs[tests_generated,:] = test
      tests_generated += 1

      # TODO: add mechanism for selecting test which is in some sense different
      #       from previous tests

      view_test(test)
      save_test(test, zeros("init_", tests_generated))

      model.log("Executing {} ({}/{})".format(test, tests_generated, model.random_init))
      test_outputs[tests_generated - 1,:] = model.sut.execute_test(test)
      model.log("Result: {}".format(test_outputs[tests_generated - 1,0]))

  # Use the tests whose fitness exceeds init_threshold as training data for the
  # critic.
  # TODO: What if the number of samples in the training data is very low? What
  #       if it's zero?
  test_critic_training = [n for n in range(tests_generated) if test_outputs[n,0] >= init_fitness_threshold]
  # TODO: We are now exiting in case we have nothing to traing with. Should we
  #       redo the initial phase if the number of samples is too low?
  if len(test_critic_training) == 0:
    model.log("No training samples found for the critic.")
    raise SystemExit

  """
  # Display/save the training data chosen for the critic.
  for i in test_critic_training:
    #view_test(test_inputs[i,:])
    save_test(test_inputs[i,:], zeros("critic_", i))
  """

  report_critic(model, test_inputs, test_outputs, test_critic_training)

  # Train the model with initial tests.
  # ---------------------------------------------------------------------------
  model.log("Training model...")
  model.train_analyzer_with_batch(test_inputs[:tests_generated,:],
                                  test_outputs[:tests_generated,:],
                                  epoch_settings=model.epoch_settings_init,
                                  log=True)
  model.train_with_batch(test_inputs[test_critic_training,:],
                         test_outputs[test_critic_training,:],
                         epoch_settings=model.epoch_settings_init,
                         log=True)

  """
  # View and save N generated tests based solely on the training on initial
  # data.
  N = 30
  new_tests = model.generate_test(N)
  for n in range(N):
    save_test(new_tests[n,:], zeros("eval_", n + 1))
  """

  """
  # Check the performance of the analyzer.
  predicted = model.predict_fitness(test_inputs[:tests_generated])
  model.log("Analyzer performance:")
  model.log("Real: Predicted:")
  for i in range(tests_generated):
    model.log("{} {}".format(test_outputs[i,0], predicted[i,0]))
  """

  # Begin the main loop for new test generation and training.
  # ---------------------------------------------------------------------------
  while tests_generated < model.N_tests:
    # Generate a new valid test with high fitness and decrease target fitness
    # as per execution of the loop.
    # -------------------------------------------------------------------------
    model.log("Starting to generate a new test.")
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

      target_fitness *= fitness_coef

      # Check if the new test has high enough fitness.
      if new_fitness >= target_fitness: break

      # View and save tests that failed to be accepted.
      #view_test(new_test)
      #save_test(new_test, zeros(zeros("failed_", tests_generated), rounds))

    # Add the new test to our test suite.
    # -------------------------------------------------------------------------
    test_inputs[tests_generated,:] = new_test
    tests_generated += 1

    model.log("Chose test {} with predicted fitness {}. Generated total {} tests of which {} were invalid.".format(new_test, new_fitness, rounds + 1, invalid))
    view_test(new_test)
    save_test(new_test, zeros("test_", tests_generated))

    # Actually run the new test on the SUT.
    model.log("Executing the test...")

    test_outputs[tests_generated - 1,:] = model.sut.execute_test(new_test)

    model.log("The actual fitness {} for the generated test.".format(test_outputs[tests_generated - 1,0]))

    # Update critic training data.
    # -------------------------------------------------------------------------
    # Now we simply add the new test if it is above the mean minus one standard
    # deviation times 0.2.
    data = test_outputs[test_critic_training,:]
    mu = data.mean(0)[0]
    sigma = data.std(0)[0]
    o = test_outputs[tests_generated - 1,0]
    if o >= mu - 0.2*sigma:
      model.log("Added the new test to the critic training data.")
      test_critic_training.append(tests_generated - 1)
      
      # Consider if we should remove the test with the lowest fitness from the
      # training data.
      if np.random.random() <= removal_probability:
        i = np.argmin(data)
        o = test_outputs[test_critic_training[i],0]
        if np.abs(o - mu) >= removal_distance:
          model.log("Removing test {} with fitness {}.".format(test_inputs[test_critic_training[i],:], o))
          test_critic_training.pop(i)
    else:
      model.log("Didn't add the new test to the critic training data.")
    del data

    report_critic(model, test_inputs, test_outputs, test_critic_training)

    # Train the model.
    # -------------------------------------------------------------------------
    model.log("Training the model...")
    model.train_analyzer_with_batch(test_inputs[:tests_generated,:],
                                    test_outputs[:tests_generated,:],
                                    epoch_settings=model.epoch_setting,
                                    log=True)
    model.train_with_batch(test_inputs[test_critic_training,:],
                           test_outputs[test_critic_training,:],
                           epoch_settings=model.epoch_settings,
                           log=True)

  # Save the trained models.
  # ---------------------------------------------------------------------------
  model.save("init", session_directory)

  # Get the final training data for the critic.
  # ---------------------------------------------------------------------------
  # Use the tests whose fitness exceeds post_threshold as the final training
  # data for the critic.
  # TODO: What if the number of samples in the training data is very low? What
  #       if it's zero?
  final_test_critic_training = [n for n in range(tests_generated) if test_outputs[n,0] >= post_fitness_threshold]
  # TODO: We are now exiting in case we have nothing to traing with. Should we
  #       redo the initial phase if the number of samples is too low?
  if len(final_test_critic_training) == 0:
    model.log("No training data for the final model.")
    raise SystemExit

  # Save the training data.
  # ---------------------------------------------------------------------------
  # Save training data.
  with open(os.path.join(session_directory, "training_data.npy"), mode="wb") as f:
    np.save(f, test_inputs)
    np.save(f, test_outputs)
    np.save(f, np.array(test_critic_training))
    np.save(f, np.array(final_test_critic_training))

  # Train the model on the complete collected data.
  # ---------------------------------------------------------------------------
  model.log("Training the final model...")
  report_critic(model, test_inputs, test_outputs, final_test_critic_training)
  model.train_with_batch(test_inputs[final_test_critic_training,:],
                         test_outputs[final_test_critic_training,:],
                         epoch_settings=model.epoch_settings_post,
                         log=True)

  # Save the final model, parameters, log, etc.
  # ---------------------------------------------------------------------------
  # Save the final model.
  model.save("final", session_directory)

  # Save training parameters.
  with open(os.path.join(session_directory, "parameters"), mode="w") as f:
    f.write(json.dumps(model.parameters))

  # Save the log.
  model.logger.save(os.path.join(session_directory, "session.log"))

  return test_inputs, test_outputs

def main_random(model_id, sut_id, model, session, session_directory, view_test, save_test, load_pregenerated_data):
  """
  Baseline random algorithm for generating a test suite.
  """

  # Format (s + "{}").format(N) with enough initial zeros.
  zeros = lambda s, N: (s + "{{:0{}d}}").format(int(log10(model.N_tests)) + 1).format(N)

  test_inputs = np.zeros(shape=(model.N_tests, model.sut.ndimensions)) # array to hold all generated tests
  test_outputs = np.zeros(shape=(model.N_tests, 1))                    # array to hold test outputs
  tests_generated = 0                                                  # how many tests are generated so far

  # Begin the main loop for new test generation.
  # ---------------------------------------------------------------------------
  while tests_generated < model.N_tests:
    model.log("Starting to generate a new test.")
    new_test = model.generate_test()

    # Add the new test to our test suite.
    # -------------------------------------------------------------------------
    test_inputs[tests_generated,:] = new_test
    tests_generated += 1

    model.log("Chose test {} with predicted fitness 1.".format(new_test))
    view_test(new_test)
    save_test(new_test, zeros("test_", tests_generated))

    # Actually run the new test on the SUT.
    model.log("Executing the test...")

    test_outputs[tests_generated - 1,:] = model.sut.execute_test(new_test)

    model.log("The actual fitness {} for the generated test.".format(test_outputs[tests_generated - 1,0]))

  # Train the model on the complete collected data.
  # ---------------------------------------------------------------------------
  # TODO

  # Save the final model, training data, log, etc.
  # ---------------------------------------------------------------------------
  # Save training parameters.
  with open(os.path.join(session_directory, "parameters"), mode="w") as f:
    f.write(json.dumps(model.parameters))

  # Save training data.
  with open(os.path.join(session_directory, "training_data.npy"), mode="wb") as f:
    np.save(f, test_inputs)
    np.save(f, test_outputs)

  # Save the log.
  model.logger.save(os.path.join(session_directory, "session.log"))

  return test_inputs, test_outputs

