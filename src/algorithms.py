#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, time, json
from math import log10

import numpy as np

from config import config

def main_ogan(model_id, sut_id, model, session, view_test, save_test, model_snapshot=False):
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
      data_X = np.load(f)
      data_Y = np.load(f)
    idx = np.random.choice(data_X.shape[0], session.random_init)
    test_inputs[:session.random_init,:] = data_X[idx,:]
    test_outputs[:session.random_init,:] = data_Y[idx,:]
    tests_generated = session.random_init
    del data_X
    del data_Y
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
    model.log("Starting to generate test {}.".format(tests_generated + 1))
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

    # Save a model snapshot if required.
    if model_snapshot:
      model.save(zeros("model_snapshot_", tests_generated), session.session_directory)

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

def main_wgan(model_id, sut_id, model, session, view_test, save_test, model_snapshot=False):
  """
  The WGAN algorithm for generating a test suite.
  """

  # Format (s + "{}").format(N) with enough initial zeros.
  zeros = lambda s, N: (s + "{{:0{}d}}").format(int(log10(session.N_tests)) + 1).format(N)

  # TODO: make configurable
  # How much to decrease the target fitness per each round when selecting a
  # new generated test.
  session.fitness_coef = 0.80
  # How many candidate tests to generate per round.
  session.N_candidate_tests = 1
  # How many buckets are used.
  session.buckets = 10
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

  # The shift for sampling training data. We increase the number a linearly
  # according to the function R.
  a0 = 0 # initial value
  a1 = 3.0 # final value
  alpha = (a1-a0)/(session.N_tests - session.random_init)
  beta = a1 - alpha*session.N_tests
  R = lambda x: alpha*x + beta
  S = lambda x: 1 / (1 + np.exp(-5*x))

  def bucket_sample(N, S, shift):
    """
    Samples N bucket indices. The distribution on the indices is defined as
    follows. Suppose that S is a nonnegative function satisfying
    S(-x) = 1 - x for all x. Consider the middle points of the buckets. We map
    the middle point of the middle bucket to 0 and the remaining middle points
    symmetrically around 0 with first middle point corresponding to -1 and the
    final to 1. We then shift these mapped middle points to the left by the
    given amount. The weight of the bucket will is S(x) where x is the mapped
    and shifted middle point.
    """

    # If the number of buckets is odd, then the middle point of the middle
    # bucket interval is mapped to 0 and otherwise the point common to the two
    # middle bucket intervals is mapped to 0.
    if session.buckets % 2 == 0:
      h = lambda x: x - (int(session.buckets/2) + 0.0)*(1/session.buckets)
    else:
      h = lambda x: x - (int(session.buckets/2) + 0.5)*(1/session.buckets)

    # We basically take the middle point of a bucket interval, map it to
    # [-1, 1] and apply S on the resulting point to find the
    # unnormalized bucket weight.
    weights = np.zeros(shape=(session.buckets))
    for n in range(session.buckets):
      weights[n] = S(h((n + 0.5)*(1/session.buckets)) - shift)
    # Normalize weights.
    weights = (weights / np.sum(weights))

    idx = np.random.choice(list(range(session.buckets)), N, p=weights)
    return idx

  def training_sample(N, X, Y, B, S, shift):
    """
    Samples N elements from X and corresponding values of Y. The sampling is
    done by picking a bucket and uniformly randomly selecting a test from the
    bucket. The probability of picking each bucket is computed via the function
    bucket_sample.
    """

    sample_X = np.zeros_like(X)
    sample_Y = np.zeros_like(Y)
    for n, bucket_idx in enumerate(bucket_sample(N, S, shift)):
      # If a bucket is empty, try one lower bucket.
      while len(B[bucket_idx]) == 0:
        bucket_idx -= 1
        bucket_idx = bucket_idx % session.buckets
      idx = np.random.choice(B[bucket_idx])
      sample_X[n] = X[idx]
      sample_Y[n] = Y[idx]

    return sample_X, sample_Y

  time_total_start = time.monotonic()

  # Generate initial tests randomly.
  # ---------------------------------------------------------------------------
  test_inputs = np.zeros(shape=(session.N_tests, model.sut.ndimensions)) # array to hold all generated tests
  test_outputs = np.zeros(shape=(session.N_tests, 1))                    # array to hold test outputs
  test_buckets = {i:[] for i in range(session.buckets)}                  # a dictionary to tell which test is in which bucket
  tests_generated = 0                                                    # how many tests are generated so far

  if session.load_pregenerated_data:
    model.log("Loading pregenerated initial tests.")
    with open(config[sut_id][model_id]["pregenerated_initial_data"], mode="br") as f:
      data_X = np.load(f)
      data_Y = np.load(f)
    idx = np.random.choice(data_X.shape[0], session.random_init)
    test_inputs[:session.random_init,:] = data_X[idx,:]
    test_outputs[:session.random_init,:] = data_Y[idx,:]
    tests_generated = session.random_init
    del data_X
    del data_Y
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

  # Assign the initial tests to buckets.
  get_bucket = lambda x: int(x*session.buckets) if x < 1.0 else session.buckets-1
  for n in range(session.random_init):
    test_buckets[get_bucket(test_outputs[n,0])].append(n)

  # Train the model with initial tests.
  # ---------------------------------------------------------------------------
  model.log("Training model...")
  time_training_start = time.monotonic()
  # Train the analyzer.
  model.train_analyzer_with_batch(test_inputs[:tests_generated,:],
                                  test_outputs[:tests_generated,:],
                                  train_settings=model.train_settings_init,
                                  log=True)
  # Train the WGAN.
  train_X, train_Y = training_sample(model.batch_size,
                                     test_inputs[:tests_generated,:],
                                     test_outputs[:tests_generated,:],
                                     test_buckets,
                                     S,
                                     R(tests_generated))
  model.train_with_batch(train_X,
                         train_Y,
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
    model.log("Starting to generate test {}.".format(tests_generated + 1))
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
    test_buckets[get_bucket(test_outputs[tests_generated - 1,0])].append(tests_generated - 1)

    model.log("The actual fitness {} for the generated test.".format(test_outputs[tests_generated - 1,0]))

    # Train the model.
    # -------------------------------------------------------------------------
    model.log("Training the model...")
    time_training_start = time.monotonic()
    # Train the analyzer.
    model.train_analyzer_with_batch(test_inputs[:tests_generated,:],
                                    test_outputs[:tests_generated,:],
                                    train_settings=model.train_settings,
                                    log=True)
    # Train the WGAN.
    train_X, train_Y = training_sample(model.batch_size,
                                       test_inputs[:tests_generated, :],
                                       test_outputs[:tests_generated, :],
                                       test_buckets,
                                       S,
                                       R(tests_generated))
    model.train_with_batch(train_X,
                           train_Y,
                           train_settings=model.train_settings,
                           log=True)
    session.time_training.append(time.monotonic() - time_training_start)

    # Save a model snapshot if required.
    if model_snapshot:
        model.save(zeros("model_snapshot_", tests_generated), session.session_directory)

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

def main_random(model_id, sut_id, model, session, view_test, save_test, model_snapshot=False):
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
    model.log("Starting to generate test {}.".format(tests_generated + 1))
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

