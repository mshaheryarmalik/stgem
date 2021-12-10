#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, time, json
from math import log10

import numpy as np

from config import config

def main_ogan(model_id, sut_id, model, session, view_test, save_test, pretrained_analyzer=False, model_snapshot=False):
  """
  The OGAN algorithm for generating a test suite.
  """

  # Format (s + "{}").format(N) with enough initial zeros.
  zeros = lambda s, N: (s + "{{:0{}d}}").format(int(log10(session.N_tests)) + 1).format(N)

  # TODO: make configurable
  # How much to decrease the target fitness per each round when selecting a
  # new generated test.
  session.fitness_coef = 0.95
  # How often to train.
  session.train_delay = 3
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

  def save_progress():
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
    time_generation_start = time.monotonic()
    invalid = 0
    while tests_generated < session.random_init:
      test = model.sut.sample_input_space()
      if model.validity(test)[0,0] == 0:
        invalid += 1
        continue
      test_inputs[tests_generated,:] = test
      tests_generated += 1

      # Save information on how many tests needed to be generated etc.
      session.time_generation.append(time.monotonic() - time_generation_start)
      session.N_tests_generated.append(invalid + 1)
      session.N_invalid_tests_generated.append(invalid)

      view_test(test)
      save_test(test, zeros("init_", tests_generated))

      model.log("Executing {} ({}/{})".format(test, tests_generated, session.random_init))
      time_execution_start = time.monotonic()
      test_outputs[tests_generated - 1,:] = model.sut.execute_test(test)
      session.time_execution.append(time.monotonic() - time_execution_start)
      model.log("Result: {}".format(test_outputs[tests_generated - 1,0]))

      time_generation_start = time.monotonic()
      invalid = 0

  # TODO: Report the quality of the initial data to session object.

  # Train the model with initial tests.
  # ---------------------------------------------------------------------------
  if (tests_generated - session.random_init) % session.train_delay == 0:
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
    best_test = None
    best_fitness = 0.0
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
      if new_fitness > best_fitness:
        best_test = new_test
        best_fitness = new_fitness

      target_fitness *= session.fitness_coef

      # Check if the new test has high enough fitness.
      if best_fitness >= target_fitness: break

    # Add the new test to our test suite.
    # -------------------------------------------------------------------------
    test_inputs[tests_generated,:] = best_test
    tests_generated += 1

    # Save information on how many tests needed to be generated etc.
    session.time_generation.append(time.monotonic() - time_generation_start)
    session.N_tests_generated.append(rounds)
    session.N_invalid_tests_generated.append(invalid)

    model.log("Chose test {} with predicted fitness {}. Generated total {} tests of which {} were invalid.".format(best_test, best_fitness, rounds + 1, invalid))
    view_test(best_test)
    save_test(best_test, zeros("test_", tests_generated))

    # Actually run the new test on the SUT.
    model.log("Executing the test...")

    time_execution_start = time.monotonic()
    test_outputs[tests_generated - 1,:] = model.sut.execute_test(best_test)
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

    # Save partial training data, logs, etc.
    save_progress()
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
  model.save("final", session.session_directory)
  save_progress()

  return test_inputs, test_outputs

def main_wgan(model_id, sut_id, model, session, view_test, save_test, pretrained_analyzer=False, model_snapshot=False):
  """
  The WGAN algorithm for generating a test suite.
  """

  # Format (s + "{}").format(N) with enough initial zeros.
  zeros = lambda s, N: (s + "{{:0{}d}}").format(int(log10(session.N_tests)) + 1).format(N)

  # TODO: make configurable
  # How much to decrease the target fitness per each round when selecting a
  # new generated test.
  session.fitness_coef = 0.95
  # How many candidate tests to generate per round.
  session.N_candidate_tests = 1
  # How often to train.
  session.train_delay = 3
  # How many bins are used.
  session.bins = 10
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
  S = lambda x: 1 / (1 + np.exp(-1*x))

  def bin_sample(N, S, shift):
    """
    Samples N bin indices. The distribution on the indices is defined as
    follows. Suppose that S is a nonnegative function satisfying
    S(-x) = 1 - x for all x. Consider the middle points of the bins. We map
    the middle point of the middle bin to 0 and the remaining middle points
    symmetrically around 0 with first middle point corresponding to -1 and the
    final to 1. We then shift these mapped middle points to the left by the
    given amount. The weight of the bin will is S(x) where x is the mapped
    and shifted middle point.
    """

    # If the number of bins is odd, then the middle point of the middle bin
    # interval is mapped to 0 and otherwise the point common to the two middle
    # bin intervals is mapped to 0.
    if session.bins % 2 == 0:
      h = lambda x: x - (int(session.bins/2) + 0.0)*(1/session.bins)
    else:
      h = lambda x: x - (int(session.bins/2) + 0.5)*(1/session.bins)

    # We basically take the middle point of a bin interval, map it to [-1, 1]
    # and apply S on the resulting point to find the unnormalized bin weight.
    weights = np.zeros(shape=(session.bins))
    for n in range(session.bins):
      weights[n] = S(h((n + 0.5)*(1/session.bins)) - shift)
    # Normalize weights.
    weights = (weights / np.sum(weights))

    idx = np.random.choice(list(range(session.bins)), N, p=weights)
    return idx

  def training_sample(N, X, Y, B, S, shift):
    """
    Samples N elements from X and corresponding values of Y. The sampling is
    done by picking a bin and uniformly randomly selecting a test from the bin,
    but we do not select the same test twice. The probability of picking each
    bin is computed via the function bin_sample.
    """

    sample_X = np.zeros(shape=(N, X.shape[1]))
    sample_Y = np.zeros(shape=(N, Y.shape[1]))
    available = {n:v.copy() for n, v in B.items()}
    for n, bin_idx in enumerate(bin_sample(N, S, shift)):
      # If a bin is empty, try one lower bin.
      while len(available[bin_idx]) == 0:
        bin_idx -= 1
        bin_idx = bin_idx % session.bins
      idx = np.random.choice(available[bin_idx])
      available[bin_idx].remove(idx)
      sample_X[n] = X[idx]
      sample_Y[n] = Y[idx]

    return sample_X, sample_Y

  def save_progress():
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

  # Load a pretrained analyzer if requested.
  if pretrained_analyzer:
    try:
      model.analyzer.load("pretrained", config[sut_id]["data_directory"])
    except Exception as E:
      print("Could not load a pretrained analyzer.")
      raise SystemExit

  time_total_start = time.monotonic()

  # Generate initial tests randomly.
  # ---------------------------------------------------------------------------
  test_inputs = np.zeros(shape=(session.N_tests, model.sut.ndimensions)) # array to hold all generated tests
  test_outputs = np.zeros(shape=(session.N_tests, 1))                    # array to hold test outputs
  test_bins = {i:[] for i in range(session.bins)}                        # a dictionary to tell which test is in which bin
  tests_generated = 0                                                    # how many tests are generated so far

  if session.load_pregenerated_data:
    # TODO: It is possible that a test is selected twice. Avoid this.
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
    time_generation_start = time.monotonic()
    invalid = 0
    while tests_generated < session.random_init:
      test = model.sut.sample_input_space()
      if model.validity(test)[0,0] == 0:
        invalid += 1
        continue
      test_inputs[tests_generated,:] = test
      tests_generated += 1

      # Save information on how many tests needed to be generated etc.
      session.time_generation.append(time.monotonic() - time_generation_start)
      session.N_tests_generated.append(invalid + 1)
      session.N_invalid_tests_generated.append(invalid)

      view_test(test)
      save_test(test, zeros("init_", tests_generated))

      model.log("Executing {} ({}/{})".format(test, tests_generated, session.random_init))
      time_execution_start = time.monotonic()
      test_outputs[tests_generated - 1,:] = model.sut.execute_test(test)
      session.time_execution.append(time.monotonic() - time_execution_start)
      model.log("Result: {}".format(test_outputs[tests_generated - 1,0]))

      time_generation_start = time.monotonic()
      invalid = 0

  # Assign the initial tests to bins.
  get_bin = lambda x: int(x*session.bins) if x < 1.0 else session.bins-1
  for n in range(session.random_init):
    test_bins[get_bin(test_outputs[n,0])].append(n)

  # Train the model with initial tests.
  # ---------------------------------------------------------------------------
  model.log("Training model...")
  time_training_start = time.monotonic()
  # Train the analyzer.
  for epoch in range(model.train_settings_init["analyzer_epochs"]):
    model.train_analyzer_with_batch(test_inputs[:tests_generated,:],
                                    test_outputs[:tests_generated,:],
                                    train_settings=model.train_settings_init,
                                    log=True)
  # Train the WGAN with different batches.
  for epoch in range(model.train_settings_init["epochs"]):
    train_X, train_Y = training_sample(min(model.batch_size, session.random_init),
                                       test_inputs[:tests_generated,:],
                                       test_outputs[:tests_generated,:],
                                       test_bins,
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
    best_test = None
    best_fitness = 0.0
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
      if new_fitness > best_fitness:
        best_test = new_test
        best_fitness = new_fitness

      target_fitness *= session.fitness_coef

      # Check if the best test has high enough fitness.
      if best_fitness >= target_fitness: break

    # Add the new test to our test suite.
    # -------------------------------------------------------------------------
    test_inputs[tests_generated,:] = best_test
    tests_generated += 1

    # Save information on how many tests needed to be generated etc.
    session.time_generation.append(time.monotonic() - time_generation_start)
    session.N_tests_generated.append(rounds*session.N_candidate_tests)
    session.N_invalid_tests_generated.append(invalid)

    model.log("Chose test {} with predicted fitness {}. Generated total {} tests of which {} were invalid.".format(best_test, best_fitness, session.N_tests_generated[-1], session.N_invalid_tests_generated[-1]))
    view_test(best_test)
    save_test(best_test, zeros("test_", tests_generated))

    # Actually run the new test on the SUT.
    model.log("Executing the test...")

    time_execution_start = time.monotonic()
    test_outputs[tests_generated - 1,:] = model.sut.execute_test(best_test)
    session.time_execution.append(time.monotonic() - time_execution_start)
    test_bins[get_bin(test_outputs[tests_generated - 1,0])].append(tests_generated - 1)

    model.log("The actual fitness {} for the generated test.".format(test_outputs[tests_generated - 1,0]))

    # Train the model.
    # -------------------------------------------------------------------------
    if (tests_generated - session.random_init) % session.train_delay == 0:
      model.log("Training the model...")
      time_training_start = time.monotonic()
      # Train the analyzer.
      analyzer_batch_size = tests_generated
      for epoch in range(model.train_settings["analyzer_epochs"]):
        """
        # We include the new tests and a number of previous tests randomly into
        # the batch.
        train_X = np.zeros(shape=(analyzer_batch_size, test_inputs.shape[1]))
        train_Y = np.zeros(shape=(analyzer_batch_size, test_outputs.shape[1]))
        for i in range(session.train_delay):
          train_X[i] = test_inputs[tests_generated - i - 1]
          train_Y[i] = test_outputs[tests_generated - i - 1]
        idx = np.random.choice(tests_generated - session.train_delay, analyzer_batch_size - session.train_delay, replace=False)
        train_X[session.train_delay:] = test_inputs[idx]
        train_Y[session.train_delay:] = test_outputs[idx]

        model.train_analyzer_with_batch(train_X,
                                        train_Y,
                                        train_settings=model.train_settings,
                                        log=True)
        """
        model.train_analyzer_with_batch(test_inputs[:tests_generated],
                                        test_outputs[:tests_generated],
                                        train_settings=model.train_settings,
                                        log=True)
      # Train the WGAN.
      for epoch in range(model.train_settings_init["epochs"]):
        # We include the new tests to the batch with high probability if and
        # only if they have high fitness.
        bin_sample(1, S, R(tests_generated))
        BS = min(model.batch_size, session.random_init)
        train_X = np.zeros(shape=(BS, test_inputs.shape[1]))
        train_Y = np.zeros(shape=(BS, test_outputs.shape[1]))
        c = 0
        for i in range(session.train_delay):
          if get_bin(test_outputs[tests_generated - i - 1]) >= bin_sample(1, S, R(tests_generated))[0]:
            train_X[c] = test_inputs[tests_generated - i - 1]
            train_Y[c] = test_outputs[tests_generated - i - 1]
            c += 1
        train_X[c:], train_Y[c:] = training_sample(BS - c,
                                                   test_inputs[:tests_generated, :],
                                                   test_outputs[:tests_generated, :],
                                                   test_bins,
                                                   S,
                                                   R(tests_generated))
        model.train_with_batch(train_X,
                               train_Y,
                               train_settings=model.train_settings,
                               log=True)
      session.time_training.append(time.monotonic() - time_training_start)

    # Save partial training data, logs, etc.
    save_progress()

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
  model.save("final", session.session_directory)
  save_progress()

  return test_inputs, test_outputs

def main_random(model_id, sut_id, model, session, view_test, save_test, pretrained_analyzer=False, model_snapshot=False):
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

  def save_progress():
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

    # Save partial training data, logs, etc.
    save_progress()

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
  save_progress()

  return test_inputs, test_outputs

