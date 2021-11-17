#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, time, json
from math import log10

import numpy as np

from config import config

def main_ogan(model_id, sut_id, model, session, view_test, save_test):
  """
  The OGAN algorithm for generating a test suite.
  """

  # Format (s + "{}").format(N) with enough initial zeros.
  zeros = lambda s, N: (s + "{{:0{}d}}").format(int(log10(session.N_tests)) + 1).format(N)

  # TODO: make configurable
  # How much to decrease the target fitness per each round when selecting a
  # new generated test.
  session.fitness_coef = 0.95
  # Stores execution times.
  session.time_total = 0
  session.time_training_total = 0
  session.time_execution_total = 0
  session.time_training = []
  session.time_generation = []
  session.time_execution = []
  # Stores how many tests needed to be generated before a test was selected.
  session.N_tests_generated = []
  # Stores how many invalid tests were generated before a test was selected.
  session.N_invalid_tests_generated = []
  # How many positive tests were generated.
  session.N_positive_tests = 0

  time_total_start = time.monotonic()

  # Generate initial tests randomly.
  # ---------------------------------------------------------------------------
  test_inputs = np.zeros(shape=(session.N_tests, model.sut.ndimensions)) # array to hold all generated tests
  test_outputs = np.zeros(shape=(session.N_tests, 1))                    # array to hold test outputs
  tests_generated = 0                                                    # how many tests are generated so far

  if session.load_pregenerated_data:
    model.log("Loading pregenerated initial tests.")
    with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
      test_inputs[:session.random_init,:] = np.load(f)[:session.random_init,:]
      test_outputs[:session.random_init,:] = np.load(f)[:session.random_init,:]
      tests_generated = session.random_init
  else:
    model.log("Generating and running {} random valid tests.".format(session.random_init))
    while tests_generated < session.random_init:
      test = model.sut.sample_input_space()
      if model.validity(test)[0,0] == 0: continue
      test_inputs[tests_generated,:] = test
      tests_generated += 1

      # TODO: add mechanism for selecting test which is in some sense different
      #       from previous tests

      view_test(test)
      save_test(test, zeros("init_", tests_generated))

      model.log("Executing {} ({}/{})".format(test, tests_generated, session.random_init))
      test_outputs[tests_generated - 1,:] = model.sut.execute_test(test)
      model.log("Result: {}".format(test_outputs[tests_generated - 1,0]))

  # TODO: Report the quality of the initial data to session object.

  # Train the model with initial tests.
  # ---------------------------------------------------------------------------
  model.log("Training model...")
  time_training_start = time.monotonic()
  model.train_with_batch(test_inputs[:tests_generated,:],
                         test_outputs[:tests_generated,:],
                         train_settings=model.train_settings_init,
                         log=True)
  session.time_training.append(time.monotonic() - time_training_start)

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
  while tests_generated < session.N_tests:
    # Generate a new valid test with high fitness and decrease target fitness
    # as per execution of the loop.
    # -------------------------------------------------------------------------
    model.log("Starting to generate a new test.")
    time_generation_start = time.monotonic()
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

      target_fitness *= session.fitness_coef

      # Check if the new test has high enough fitness.
      if new_fitness >= target_fitness: break

    # Add the new test to our test suite.
    # -------------------------------------------------------------------------
    test_inputs[tests_generated,:] = new_test
    tests_generated += 1

    # Save information on how many tests needed to be generated etc.
    session.time_generation.append(time.monotonic() - time_generation_start)
    session.N_tests_generated.append(rounds)
    session.N_invalid_tests_generated.append(invalid)

    model.log("Chose test {} with predicted fitness {}. Generated total {} tests of which {} were invalid.".format(new_test, new_fitness, rounds + 1, invalid))
    view_test(new_test)
    save_test(new_test, zeros("test_", tests_generated))

    # Actually run the new test on the SUT.
    model.log("Executing the test...")

    time_execution_start = time.monotonic()
    test_outputs[tests_generated - 1,:] = model.sut.execute_test(new_test)
    session.time_execution.append(time.monotonic() - time_execution_start)

    model.log("The actual fitness {} for the generated test.".format(test_outputs[tests_generated - 1,0]))

    # Train the model.
    # -------------------------------------------------------------------------
    model.log("Training the model...")
    time_training_start = time.monotonic()
    model.train_with_batch(test_inputs[:tests_generated,:],
                           test_outputs[:tests_generated,:],
                           train_settings=model.train_settings,
                           log=True)
    session.time_training.append(time.monotonic() - time_training_start)

  # Train the model on the complete collected data.
  # ---------------------------------------------------------------------------
  # TODO

  # Record some information for saving.
  # ---------------------------------------------------------------------------
  session.N_positive_tests = int(sum(test_outputs >= model.sut.target)[0])
  session.fitness_avg = np.mean(test_outputs)
  session.fitness_std = np.std(test_outputs)
  session.time_total = (time.monotonic() - time_total_start)
  session.time_training_total = sum(session.time_training)
  session.time_execution_total = sum(session.time_execution)

  # Save everything
  # ---------------------------------------------------------------------------
  # Save the trained models.
  model.save("init", session.session_directory)
  #model.save("final", session.session_directory)

  # Save the training data.
  with open(os.path.join(session.session_directory, "training_data.npy"), mode="wb") as f:
    np.save(f, test_inputs)
    np.save(f, test_outputs)

  # Save training parameters.
  with open(os.path.join(session.session_directory, "parameters"), mode="w") as f:
    f.write(json.dumps(model.parameters))

  # Save session parameters.
  with open(os.path.join(session.session_directory, "session_parameters"), mode="w") as f:
    f.write(json.dumps(session.parameters))

  # Save the log.
  model.logger.save(os.path.join(session.session_directory, "session.log"))

  return test_inputs, test_outputs

def main_wgan(model_id, sut_id, model, session, view_test, save_test):
  """
  The WGAN algorithm for generating a test suite.
  """

  # Format (s + "{}").format(N) with enough initial zeros.
  zeros = lambda s, N: (s + "{{:0{}d}}").format(int(log10(session.N_tests)) + 1).format(N)

  # TODO: make configurable
  # How much to decrease the target fitness per each round when selecting a
  # new generated test.
  session.fitness_coef = 0.95
  # Include tests whose fitness exceed the threshould to the initial critic
  # training data.
  session.init_fitness_threshold = 0.1
  # Include tests whose fitness exceed the threshould to the final critic
  # training data.
  session.post_fitness_threshold = 0.5
  # How many candidate tests to generate per round.
  session.N_candidate_tests = 1
  # The interval where the probability to remove a sample from the critic
  # training data lies.
  session.removal_probability_1 = 0.5
  session.removal_probability_2 = 0.8
  # Stores execution times.
  session.time_total = 0
  session.time_training_total = 0
  session.time_execution_total = 0
  session.time_training = []
  session.time_generation = []
  session.time_execution = []
  # Stores how many tests needed to be generated before a test was selected.
  session.N_tests_generated = []
  # Stores how many invalid tests were generated before a test was selected.
  session.N_invalid_tests_generated = []
  # How many positive tests were generated.
  session.N_positive_tests = 0

  # Define the function which returns the removal probability from critic
  # training data.
  # Currently the idea is to multiply the difference
  # removal_probability_2 - removal_probability_1 by a sigmoid value which is
  # computed from the current number of tests generated as follows:
  #   - the mapping is linear,
  #   - when 70 % of the tests after the initial tests have are generated, the
  #   - sigmoid value is -2,
  #   - when 70 % of the tests after the initial tests have are generated, the
  #   - sigmoid value is 1,
  S = lambda x: 1 / (1 + np.exp(-x))
  alpha = (2 - (-1)) / (0.2 * (session.N_tests - session.random_init))
  beta = 2 - session.random_init * alpha - (2 - (-1))*(0.9/0.2)
  R = lambda x: session.removal_probability_1 + (session.removal_probability_2 - session.removal_probability_1) * S(alpha * x + beta)

  def report_critic(model, test_inputs, test_outputs, test_critic_training):
    """
    Report some statistics on the critic training data.
    """

    data = test_outputs[test_critic_training,:]
    mu = data.mean(0)[0]
    sigma = data.std(0)[0]
    model.log("Critic training data has {} samples with mean {} and std {}.".format(len(test_critic_training), mu, sigma))

  time_total_start = time.monotonic()

  # Generate initial tests randomly.
  # ---------------------------------------------------------------------------
  test_inputs = np.zeros(shape=(session.N_tests, model.sut.ndimensions)) # array to hold all generated tests
  test_outputs = np.zeros(shape=(session.N_tests, 1))                    # array to hold test outputs
  tests_generated = 0                                                  # how many tests are generated so far
  test_critic_training = []                                            # list which indicates which rows of test_inputs belong to the critic training data

  if session.load_pregenerated_data:
    model.log("Loading pregenerated initial tests.")
    with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
      test_inputs[:session.random_init,:] = np.load(f)[:session.random_init,:]
      test_outputs[:session.random_init,:] = np.load(f)[:session.random_init,:]
      tests_generated = session.random_init
  else:
    model.log("Generating and running {} random valid tests.".format(session.random_init))
    while tests_generated < session.random_init:
      test = model.sut.sample_input_space()
      if model.validity(test)[0,0] == 0: continue
      test_inputs[tests_generated,:] = test
      tests_generated += 1

      view_test(test)
      save_test(test, zeros("init_", tests_generated))

      model.log("Executing {} ({}/{})".format(test, tests_generated, session.random_init))
      test_outputs[tests_generated - 1,:] = model.sut.execute_test(test)
      model.log("Result: {}".format(test_outputs[tests_generated - 1,0]))

  # TODO: Report the quality of the initial data to session object.

  # Use the tests whose fitness exceeds init_threshold as training data for the
  # critic.
  # TODO: What if the number of samples in the training data is very low? What
  #       if it's zero?
  test_critic_training = [n for n in range(tests_generated) if test_outputs[n,0] >= session.init_fitness_threshold]
  # TODO: We are now exiting in case we have nothing to traing with. Should we
  #       redo the initial phase if the number of samples is too low?
  if len(test_critic_training) == 0:
    model.log("No training samples found for the critic.")
    raise SystemExit
  session.critic_training_data_history = [test_critic_training.copy()]

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
  time_training_start = time.monotonic()
  model.train_analyzer_with_batch(test_inputs[:tests_generated,:],
                                  test_outputs[:tests_generated,:],
                                  train_settings=model.train_settings_init,
                                  log=True)
  model.train_with_batch(test_inputs[test_critic_training,:],
                         test_outputs[test_critic_training,:],
                         train_settings=model.train_settings_init,
                         log=True)
  session.time_training.append(time.monotonic() - time_training_start)

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
  while tests_generated < session.N_tests:
    # Generate a new valid test with high fitness and decrease target fitness
    # as per execution of the loop.
    # -------------------------------------------------------------------------
    model.log("Starting to generate a new test.")
    time_generation_start = time.monotonic()
    target_fitness = 1
    rounds = 0
    invalid = 0
    while True:
      # Gererate several tests and pick the one with best predicted fitness.
      candidate_tests = model.generate_test(session.N_candidate_tests)
      rounds += 1

      # Pick only the valid tests.
      candidate_tests = candidate_tests[(model.validity(candidate_tests) == 1.0).reshape(-1)]
      invalid += session.N_candidate_tests - candidate_tests.shape[0]
      if candidate_tests.shape[0] == 0:
        continue

      # Find best test.
      tests_predicted_fitness = model.predict_fitness(candidate_tests)
      idx = np.argmax(tests_predicted_fitness)
      new_test = candidate_tests[idx].reshape(1, -1)
      new_fitness = tests_predicted_fitness[idx][0]

      target_fitness *= session.fitness_coef

      # Check if the new test has high enough fitness.
      if new_fitness >= target_fitness: break

    # Add the new test to our test suite.
    # -------------------------------------------------------------------------
    test_inputs[tests_generated,:] = new_test
    tests_generated += 1

    # Save information on how many tests needed to be generated etc.
    session.time_generation.append(time.monotonic() - time_generation_start)
    session.N_tests_generated.append(rounds*session.N_candidate_tests)
    session.N_invalid_tests_generated.append(invalid)

    model.log("Chose test {} with predicted fitness {}. Generated total {} tests of which {} were invalid.".format(new_test, new_fitness, session.N_tests_generated[-1], session.N_invalid_tests_generated[-1]))
    view_test(new_test)
    save_test(new_test, zeros("test_", tests_generated))

    # Actually run the new test on the SUT.
    model.log("Executing the test...")

    time_execution_start = time.monotonic()
    test_outputs[tests_generated - 1,:] = model.sut.execute_test(new_test)
    session.time_execution.append(time.monotonic() - time_execution_start)

    model.log("The actual fitness {} for the generated test.".format(test_outputs[tests_generated - 1,0]))

    # Update critic training data.
    # -------------------------------------------------------------------------
    # We always add the new test to critic training data if it is better than
    # the worst test.
    idx = np.argmin(test_outputs[test_critic_training,:])
    worst_fitness = test_outputs[test_critic_training,:][idx][0]
    if test_outputs[tests_generated - 1,0] >= worst_fitness:
      model.log("Added the new test to the critic training data.")
      test_critic_training.append(tests_generated - 1)
    else:
      model.log("Didn't add the new test to the critic training data.")

    # Consider if we should remove the worst test.
    removal_probability = R(tests_generated)
    model.log("Current removal probability: {}".format(removal_probability))
    if np.random.random() >= removal_probability:
      # We do not remove the worst test if its fitness deviates less that 0.1
      # from the mean of the tests in the critic training data. This is mainly
      # to prevent removing tests with fitness 1 when the fitness is binary
      # with values 0 and 1.
      mean = test_outputs[test_critic_training,:].mean()
      if mean - worst_fitness > 0.1:
        model.log("Removing test {} with worst fitness {}.".format(test_inputs[idx, :], worst_fitness))
        test_critic_training.pop(idx)

    session.critic_training_data_history.append(test_critic_training.copy())
    report_critic(model, test_inputs, test_outputs, test_critic_training)

    # Train the model.
    # -------------------------------------------------------------------------
    model.log("Training the model...")
    time_training_start = time.monotonic()
    model.train_analyzer_with_batch(test_inputs[:tests_generated,:],
                                    test_outputs[:tests_generated,:],
                                    train_settings=model.train_settings,
                                    log=True)
    model.train_with_batch(test_inputs[test_critic_training,:],
                           test_outputs[test_critic_training,:],
                           train_settings=model.train_settings,
                           log=True)
    session.time_training.append(time.monotonic() - time_training_start)

  # Get the final training data for the critic.
  # ---------------------------------------------------------------------------
  # Use the tests whose fitness exceeds post_threshold as the final training
  # data for the critic.
  # TODO: What if the number of samples in the training data is very low? What
  #       if it's zero?
  final_test_critic_training = [n for n in range(tests_generated) if test_outputs[n,0] >= session.post_fitness_threshold]
  # TODO: We are now exiting in case we have nothing to traing with. Should we
  #       redo the initial phase if the number of samples is too low?
  if len(final_test_critic_training) == 0:
    model.log("No training data for the final model.")

  # Train the model on the complete collected data.
  # ---------------------------------------------------------------------------
  model.log("Training the final model...")
  report_critic(model, test_inputs, test_outputs, final_test_critic_training)
  model.train_with_batch(test_inputs[final_test_critic_training,:],
                         test_outputs[final_test_critic_training,:],
                         train_settings=model.train_settings_post,
                         log=True)

  # Record some information for saving.
  # ---------------------------------------------------------------------------
  session.N_positive_tests = int(sum(test_outputs >= model.sut.target)[0])
  session.fitness_avg = np.mean(test_outputs)
  session.fitness_std = np.std(test_outputs)
  session.time_total = (time.monotonic() - time_total_start)
  session.time_training_total = sum(session.time_training)
  session.time_execution_total = sum(session.time_execution)

  # Save everything
  # ---------------------------------------------------------------------------
  # Save the trained models.
  model.save("init", session.session_directory)
  model.save("final", session.session_directory)

  # Save the training data.
  with open(os.path.join(session.session_directory, "training_data.npy"), mode="wb") as f:
    np.save(f, test_inputs)
    np.save(f, test_outputs)
    np.save(f, np.array(test_critic_training))
    np.save(f, np.array(final_test_critic_training))

  # Save training parameters.
  with open(os.path.join(session.session_directory, "parameters"), mode="w") as f:
    f.write(json.dumps(model.parameters))

  # Save session parameters.
  with open(os.path.join(session.session_directory, "session_parameters"), mode="w") as f:
    f.write(json.dumps(session.parameters))

  # Save the log.
  model.logger.save(os.path.join(session.session_directory, "session.log"))

  return test_inputs, test_outputs

def main_random(model_id, sut_id, model, session, view_test, save_test):
  """
  Baseline random algorithm for generating a test suite.
  """

  # Format (s + "{}").format(N) with enough initial zeros.
  zeros = lambda s, N: (s + "{{:0{}d}}").format(int(log10(session.N_tests)) + 1).format(N)

  # Stores execution times.
  session.time_total = 0
  session.time_training_total = 0
  session.time_execution_total = 0
  session.time_training = []
  session.time_generation = []
  session.time_execution = []
  # Stores how many tests needed to be generated before a test was selected.
  session.N_tests_generated = []
  # Stores how many invalid tests were generated before a test was selected.
  session.N_invalid_tests_generated = []
  # How many positive tests were generated.
  session.N_positive_tests = 0

  time_total_start = time.monotonic()

  test_inputs = np.zeros(shape=(session.N_tests, model.sut.ndimensions)) # array to hold all generated tests
  test_outputs = np.zeros(shape=(session.N_tests, 1))                    # array to hold test outputs
  tests_generated = 0                                                  # how many tests are generated so far

  # Begin the main loop for new test generation.
  # ---------------------------------------------------------------------------
  while tests_generated < session.N_tests:
    model.log("Starting to generate a new test.")
    time_generation_start = time.monotonic()
    rounds = 0
    invalid = 0
    while True:
      new_test = model.generate_test()
      rounds += 1

      # Check if the test is valid.
      if model.validity(new_test)[0,0] == 0:
        invalid += 1
      else:
        break

    # Add the new test to our test suite.
    # -------------------------------------------------------------------------
    test_inputs[tests_generated,:] = new_test
    tests_generated += 1

    # Save information on how many tests needed to be generated etc.
    session.time_generation.append(time.monotonic() - time_generation_start)
    session.N_tests_generated.append(rounds)
    session.N_invalid_tests_generated.append(invalid)

    model.log("Chose test {} with predicted fitness 1.".format(new_test))
    view_test(new_test)
    save_test(new_test, zeros("test_", tests_generated))

    # Actually run the new test on the SUT.
    model.log("Executing the test...")

    time_execution_start = time.monotonic()
    test_outputs[tests_generated - 1,:] = model.sut.execute_test(new_test)
    session.time_execution.append(time.monotonic() - time_execution_start)

    model.log("The actual fitness {} for the generated test.".format(test_outputs[tests_generated - 1,0]))

  # Record some information for saving.
  # ---------------------------------------------------------------------------
  session.N_positive_tests = int(sum(test_outputs >= model.sut.target)[0])
  session.fitness_avg = np.mean(test_outputs)
  session.fitness_std = np.std(test_outputs)
  session.time_total = (time.monotonic() - time_total_start)
  session.time_training_total = sum(session.time_training)
  session.time_execution_total = sum(session.time_execution)
 
  # Save everything
  # ---------------------------------------------------------------------------
  # Save training data.
  with open(os.path.join(session.session_directory, "training_data.npy"), mode="wb") as f:
    np.save(f, test_inputs)
    np.save(f, test_outputs)

  # Save training parameters.
  with open(os.path.join(session.session_directory, "parameters"), mode="w") as f:
    f.write(json.dumps(model.parameters))

  # Save session parameters.
  with open(os.path.join(session.session_directory, "session_parameters"), mode="w") as f:
    f.write(json.dumps(session.parameters))

  # Save the log.
  model.logger.save(os.path.join(session.session_directory, "session.log"))

  return test_inputs, test_outputs

