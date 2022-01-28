#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

class Model_WOGAN:
  pass

class WOGAN(Algorithm):
  """
  Parameters:
  -----------
  N_tests: The total number of tests generated. Needed only for getting certain
           parameters right.
  N_random_init: How many initial random tests are generated.
  use_predefined_random_data: Whether or not use pregenerated random data for
                              the initial random generation. If there is more
                              pregenerated data than needed, we select always
                              the initial data for consistency. It is up to the
                              caller to ensure randomization if desired.
  predefined_random_data: X, Y
  fitness_coef: How much to decrease the target fitness per each round when
                selecting a new test.
  N_candidate_tests: How many candidate tests to generate per round.
  train_delay: How often to train a model meaning how many tests need to be
               generated before we train again.
  bins: How many bins are used.
  shift_function: Which predefined shift function to use. Values 'linear'.
  shift_function_parameters: Define the parameters of the shift function.
                             linear: initial: Initial value.
                                     final: Final value.
  """


  def __init__(self, sut, objective, parameters):
    super().__init__(sut, objective, )
    self.parameter_names = ["N_tests",
                            "N_random_init",
                            "use_predefined_random_data",
                            "predefined_random_data",
                            "fitness_coef",
                            "N_candidate_tests",
                            "train_delay",
                            "bins",
                            "shift_function",
                            "shift_function_parameters"]
    self.parameters = parameters
    self.log = lambda s: self._log("algorithm", s)

    self.get_bin = lambda x: int(x*self.bins) if x < 1.0 else self.bins-1

    # Setup the objective function.
    # -------------------------------------------------------------------------

    # Setup the models.
    # -------------------------------------------------------------------------

    # Setup the shift function for sampling training data.
    # -------------------------------------------------------------------------
    if self.shift_function is None:
      raise Exception("No shift function defined.")
    if self.shift_function_parameters is None:
      raise Exception("No shift function parameters defined.")

    if self.shift_function == "linear":
      # We increase the shift linearly according to the given initial and given
      # final value.
      alpha = (a1-a0)/(self.N_tests - self.N_random_init)
      beta = a1 - alpha*self.N_tests
      self.shift = lambda x: alpha*x + beta
    else:
      raise Exception("No shift function type '{}'.".format(self.shift_function))

    # Setup the function for computing the bin weights.
    # -------------------------------------------------------------------------
    self.bin_weight = lambda x: 1 / (1 + np.exp(-1*x))

  def bin_sample(self, N, shift):
    """
    Samples N bin indices.
    """

    """
    The distribution on the indices is defined as follows. Suppose that S is a
    nonnegative function satisfying S(-x) = 1 - x for all x. Consider the
    middle points of the bins. We map the middle point of the middle bin to 0
    and the remaining middle points symmetrically around 0 with first middle
    point corresponding to -1 and the final to 1. We then shift these mapped
    middle points to the right by the given amount. The weight of the bin will
    is S(x) where x is the mapped and shifted middle point. We use the function
    self.bin_weight for S.
    """

    # If the number of bins is odd, then the middle point of the middle bin
    # interval is mapped to 0 and otherwise the point common to the two middle
    # bin intervals is mapped to 0.
    # TODO: This could be predefined in __init__.
    if self.bins % 2 == 0:
      h = lambda x: x - (int(self.bins/2) + 0.0)*(1/self.bins)
    else:
      h = lambda x: x - (int(self.bins/2) + 0.5)*(1/self.bins)

    # We basically take the middle point of a bin interval, map it to [-1, 1]
    # and apply S on the resulting point to find the unnormalized bin weight.
    weights = np.zeros(shape=(self.bins))
    for n in range(session.bins):
      weights[n] = self.bin_weight(h((n + 0.5)*(1/self.bins)) - shift)
    # Normalize weights.
    weights = (weights / np.sum(weights))

    idx = np.random.choice(list(range(self.bins)), N, p=weights)
    return idx

  def training_sample(N, X, Y, B, shift):
    """
    Samples N elements from X. The sampling is done by picking a bin and
    uniformly randomly selecting a test from the bin, but we do not select the
    same test twice. The probability of picking each bin is computed via the
    function bin_sample.
    """

    sample_X = np.zeros(shape=(N, X.shape[1]))
    available = {n:v.copy() for n, v in B.items()}
    for n, bin_idx in enumerate(self.bin_sample(N, shift)):
      # If a bin is empty, try one lower bin.
      while len(available[bin_idx]) == 0:
        bin_idx -= 1
        bin_idx = bin_idx % self.bins
      idx = np.random.choice(available[bin_idx])
      available[bin_idx].remove(idx)
      sample_X[n] = X[idx]

    return sample_X

  def generate_test(self):
    self.timer_start("total")

    # Generate initial tests randomly.
    # -------------------------------------------------------------------------
    test_inputs = np.zeros(shape=(self.N_tests, self.sut.ndimensions))         # array to hold all generated tests
    test_outputs = np.zeros(shape=(self.N_tests, self.objective.dim))          # array to hold test outputs
    test_bins = [{i:[] for i in range(self.bins)} for m in self.objective.dim] # a dictionary to tell which test is in which bin for each model
    model_trained = [0 for m in self.objective.dim]                            # keeps track how many tests were generated when a model was previously trained
    tests_generated = 0                                                        # how many tests have been generated so far

    if self.use_predefined_random_data:
      self.timer_start("generation")
      self.log("Loading {} predefined random tests.".format(self.N_random_init))
      # TODO: It should be the callers headache to see that there are no duplicate tests.
      if self.predefined_random_data["test_inputs"].shape[0] < self.N_random_init:
        raise Exception("Only {} predefined random tests given while {} expected.".format(self.predefined_random_data["test_inputs"].shape[0], self.N_random_init))
      if self.predefined_random_data["test_outputs"].shape[0] < self.N_random_init:
        raise Exception("Only {} predefined random test outputs given while {} expected.".format(self.predefined_random_data["test_outputs"].shape[0], self.N_random_init))
      if self.predefined_random_data["test_outputs"].shape[1] != self.objective.dim:
        raise Exception("Expected {} test outputs while {} were provided.".format(self.objective.dim, self.predefined_random_data["test_outputs"].shape[1]))
      for i in range(self.N_random_init):
        test_inputs[i,:] = self.predefined_random_data["test_inputs"][i,:]
        output = self.predefined_random_data["test_outputs"][i,:]
        test_outputs[i,:] = output
        self.objective_selector.update(np.argmax(output))

        tests_generated += 1
        self.save_history("generation_time", self.timer_reset("generation"))
        self.save_history("N_tests_generated", 1)
        self.save_history("N_invalid_tests_generated", 0)
        self.save_history("execution_time", 0)

        self.timers_hold()
        yield test_inputs[i], test_outputs[i]
        self.timers_resume()

        self.timer_start("generation")
    else:
      # TODO: Same test could be generated twice, but this is unlikely.
      self.timer_start("generation")
      self.log("Generating and running {} random valid tests.".format(self.N_random_init))
      invalid = 0
      while tests_generated < self.N_random_init:
        test = self.sut.sample_input_space()
        if self.sut.validity(test) == 0:
          invalid += 1
          continue
        test_inputs[tests_generated,:] = test
        tests_generated += 1

        # Save information on how many tests needed to be generated etc.
        self.save_history("generation_time", self.timer_reset("generation"))
        self.save_history("N_tests_generated", invalid + 1)
        self.save_history("N_invalid_tests_generated", invalid)

        self.log("Executing {} ({}/{})".format(test, tests_generated, self.N_random_init))
        self.timer_start("execution")
        test_output = self.objective(self.sut.execute_test(test))
        test_outputs[tests_generated - 1,:] = test_output
        self.objective_selector.update(np.argmax(test_output))
        self.save_history("execution_time", self.timer_reset("execution"))
        self.log("Result: {}".format(test_outputs[tests_generated - 1,0]))

        self.timers_hold()
        yield test, test_output
        self.timers_resume()

        self.timer_start("generation")
        invalid = 0

    self.timer_reset("generation")

    # Assign the initial tests to bins.
    for i in range(self.objective.dim):
      for j in range(self.N_random_init):
        test_bins[i][self.get_bin(test_outputs[j,i])].append(j)

    # Train the models with the initial tests.
    # -------------------------------------------------------------------------
    # Notice that in principle we should train all models here. We however opt
    # to train only active models for more generality. It is up to the caller
    # to ensure that all models are trained here if so desired.
    self.timer_start("training")
    for i in self.objective_selector.select():
      self.model.log("Training analyzer {}...".format(i + 1))
      # Train the analyzer.
      for epoch in range(self.models[i].train_settings_init["analyzer_epochs"]):
        self.models[i].train_analyzer_with_batch(test_inputs[:tests_generated,:],
                                                 test_outputs[:tests_generated,:],
                                                 train_settings=self.models[i].analyzer_train_settings_init,
                                                 log=True)
      # Train the WGAN.
      self.model.log("Training WGAN model {}...".format(i + 1))
      for epoch in range(self.models[i].train_settings_init["epochs"]):
        train_X = self.training_sample(min(self.models[i].batch_size, self.N_random_init),
                                       test_inputs[:tests_generated,:],
                                       test_outputs[:tests_generated,:],
                                       test_bins[i],
                                       self.shift(tests_generated))
        self.models[i].train_with_batch(train_X,
                                        train_settings=self.models[i].train_settings_init,
                                        log=True)
      self.model_trained[i] = tests_generated
    self.save_history("training_time", self.timer_reset("training"))

    # Begin the main loop for new test generation and training.
    # -------------------------------------------------------------------------
    while True:
      # We generate a new valid test as follows. For each active model, we
      # generate new tests using the model, discard invalid tests, and estimate
      # the objective function values using the analyzer. The test with the
      # highest objective function component is treated as the candidate, and
      # the candidate is accepted if the value exceeds the target. We decrease
      # the target fitness as per execution of the loop. We use a prioprity
      # queue to track the best tests in case that an estimated good test was
      # generated just before the target threshold was lowered enough for it to
      # be selected.
      # -----------------------------------------------------------------------
      self.timer_start("generation")
      self.log("Starting to generate test {}.".format(tests_generated + 1))
      heap = []
      target_fitness = 1
      rounds = 0
      invalid = 0
      active_models = self.objective_selector.select()
      while True:
        # TODO: Avoid selecting similar or same tests.
        rounds += 1
        for i in active_models:
          while True:
            # Generate several tests and pick the one with best predicted
            # objective function component. We do this as long as we find at
            # least one valid test.
            candidate_tests = self.models[i].generate_test(self.N_candidate_tests)

            # Pick only the valid tests.
            valid_idx = [i for i in range(self.N_candidate_tests) if self.sut.validity(candidate_tests[i]) == 1]
            candidate_tests = candidate_tests[valid_idx]
            invalid += self.N_candidate_tests - len(valid_idx)
            if candidate_tests.shape[0] == 0:
              continue

            # Estimate test fitnesses and add them to heap.
            tests_predicted_objective = self.sut.predict_objective(candidate_tests)
            for j in range(tests_predicted_objective.shape[0]):
              heapq.heappush(heap, (1 - tests_predicted_fitness[j,0], candidate_tests[j]))

            break

        target_fitness *= self.fitness_coef

        # Check if the best predicted test is good enough.
        if 1 - heap[0][0] >= target_fitness: break

      # Add the new test to our test suite.
      # -------------------------------------------------------------------------
      best_test = heap[0][1]
      best_fitness = 1 - heap[0][0]
      test_inputs[tests_generated,:] = best_test
      tests_generated += 1

      # Save information on how many tests needed to be generated etc.
      self.save_history("generation_time", self.timer_reset("generation"))
      N_generated = rounds*self.N_candidate_tests
      self.save_history("N_tests_generated", N_generated)
      self.save_history("N_invalid_tests_generated", invalid)

      self.log("Chose test {} with predicted fitness {}. Generated total {} tests of which {} were invalid.".format(best_test, best_fitness, self.N_generated, invalid))

      # Actually run the new test on the SUT.
      self.log("Executing the test...")

      self.timer_start("execution")
      output = self.objective(self.sut.execute_test(best_test))
      test_outputs[tests_generated - 1,:] = output
      self.save_history("execution_time", self.timer_reset("execution"))
      # Place the test into appropriate bin for each model.
      for i in range(self.objective.dim):
        test_bins[i][self.get_bin(test_outputs[tests_generated - 1,i])].append(tests_generated - 1)

      # Update which model achieved best objective function component.
      self.objective_selector.update(np.argmax(output))

      self.log("The actual fitness {} for the generated test.".format(test_outputs[tests_generated - 1,0]))

      self.timers_hold()
      yield best_test, output
      self.timers_resume()

      # Train the models.
      # -----------------------------------------------------------------------
      # We train the models which were involved in the test generation. We do
      # not take the updated active model into account. We train only if enough
      # delay since the last training.
      self.timer_start("training")
      for i in active_models:
        if tests_generated - model_trained[i] >= self.train_delay:
          self.model.log("Training analyzer {}...".format(i+1))
          # TODO: Clean up the below code.
          analyzer_batch_size = tests_generated
          for epoch in range(self.models[i].train_settings["analyzer_epochs"]):
            """
            # We include the new tests and a number of previous tests randomly into
            # the batch.
            train_X = np.zeros(shape=(analyzer_batch_size, test_inputs.shape[1]))
            train_Y = np.zeros(shape=(analyzer_batch_size, test_outputs.shape[1]))
            for i in range(self.train_delay):
              train_X[i] = test_inputs[tests_generated - i - 1]
              train_Y[i] = test_outputs[tests_generated - i - 1]
            idx = np.random.choice(tests_generated - self.train_delay, analyzer_batch_size - self.train_delay, replace=False)
            train_X[self.train_delay:] = test_inputs[idx]
            train_Y[self.train_delay:] = test_outputs[idx]

            self.models[i].train_analyzer_with_batch(train_X,
                                                     train_Y,
                                                     train_settings=self.models[i].train_settings,
                                                     log=True)
            """
            self.models[i].train_analyzer_with_batch(test_inputs[:tests_generated],
                                                     test_outputs[:tests_generated],
                                                     train_settings=self.models[i].train_settings,
                                                     log=True)

          # Train the WGAN.
          self.log("Training the WGAN model {}...".format(i+1))
          for epoch in range(self.models[i].train_settings_init["epochs"]):
            # We include the new tests to the batch with high probability if
            # and only if they have high fitness.
            BS = min(self.models[i].batch_size, self.N_random_init)
            train_X = np.zeros(shape=(BS, test_inputs.shape[1]))
            train_Y = np.zeros(shape=(BS, test_outputs.shape[1]))
            c = 0
            for j in range(self.train_delay):
              if self.get_bin(test_outputs[tests_generated - j - 1,i]) >= self.bin_sample(1, R(tests_generated))[0]:
                train_X[c] = test_inputs[tests_generated - j - 1]
                train_Y[c] = test_outputs[tests_generated - j - 1,i]
                c += 1
            train_X[c:] = self.training_sample(BS - c,
                                               test_inputs[:tests_generated, :],
                                               test_outputs[:tests_generated, :],
                                               test_bins[i],
                                               R(tests_generated))
            self.models[i].train_with_batch(train_X,
                                            train_settings=self.model[i].train_settings,
                                            log=True)
      self.save_history("training_time", self.timer_reset("training"))

    # Record some information.
    # ---------------------------------------------------------------------------
    # TODO: Currently this code is never reached. Where to put it?
    self.save_history("N_positive_tests", int(sum(test_outputs >= model.sut.target)[0]), single=True)
    self.save_history("fitness_avg", np.mean(test_outputs), single=True)
    self.save_history("fitness_sd", np.std(test_outputs), single=True)
    self.save_history("time_total", self.timer_reset("total"), single=True)
    self.save_history("time_training_total", sum(self.get_history("training_time")), single=True)
    self.save_history("time_generation_total", sum(self.get_history("generation_time")), single=True)
    self.save_history("time_execution_total", sum(self.get_history("execution_time")), single=True)

