#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib, heapq

import numpy as np

from algorithm.algorithm import Algorithm

class OGAN(Algorithm):
    """
    Implements the online generative adversarial network algorithm.
    """

    def __init__(self, sut, test_repository, objective_func, objective_selector, parameters, logger=None):
        super().__init__(sut, test_repository, objective_func, objective_selector, logger)
        self.parameters = parameters

        # Setup the models.
        # ---------------------------------------------------------------------
        # Load the specified OGAN model and initialize it.
        module = importlib.import_module(".model", "algorithm.ogan")
        model_class = getattr(module, self.parameters["ogan_model"])
        self.models = [model_class(sut=self.sut, parameters=parameters, logger=logger) for _ in range(self.objective_func.dim)]

    def generate_test(self):
        self.timer_start("total")

        tests_generated = 0                                         # how many tests have been generated so far
        model_trained = [0 for m in range(self.objective_func.dim)] # keeps track how many tests were generated when a model was previously trained

        # Generate initial tests randomly.
        # ---------------------------------------------------------------------
        for idx in self._generate_initial_random_tests():
            yield idx
        tests_generated = len(self.test_suite)

        # TODO: Add check that we get at least a couple samples for training.
        #       Otherwise we get a cryptic error.

        # Train the models with the initial tests.
        # ---------------------------------------------------------------------
        # Notice that in principle we should train all models here. We however
        # opt to train only active models for more generality. It is up to the
        # caller to ensure that all models are trained here if so desired.
        self.timer_start("training")
        for i in self.objective_selector.select():
            self.log("Training the OGAN model {}...".format(i + 1))
            dataX, dataY = self.test_repository.get(self.test_suite)
            dataX = np.array(dataX)
            dataY = np.array(dataY)
            for epoch in range(self.models[i].train_settings_init["epochs"]):
                self.models[i].train_with_batch(dataX,
                                                dataY,
                                                train_settings=self.models[i].train_settings_init
                                               )
            model_trained[i] = tests_generated
        self.save_history("training_time", self.timer_reset("training"))

        # Begin the main loop for new test generation and training.
        # ---------------------------------------------------------------------
        while True:
            # We generate a new valid test as follows. For each active model,
            # we generate new tests using the model, discard invalid tests, and
            # estimate the objective function values using the discriminator.
            # The test with the highest objective function component is treated
            # as the candidate, and the candidate is accepted if the value
            # exceeds the target. We decrease the target fitness as per
            # execution of the loop. We use a prioprity queue to track the best
            # tests in case that an estimated good test was generated just
            # before the target threshold was lowered enough for it to be
            # selected.
            self.timer_start("generation")
            self.log("Starting to generate test {}.".format(tests_generated + 1))
            heap = []
            target_fitness = 1
            entry_count = 0 # this is to avoid comparing tests when two tests added to the heap have the same predicted objective
            rounds = 0
            invalid = 0
            active_models = self.objective_selector.select()
            while True:
                # TODO: Avoid selecting similar or same tests.
                rounds += 1
                for i in active_models:
                    while True:
                        # Generate several tests and pick the one with best
                        # predicted objective function component. We do this as
                        # long as we find at least one valid test.
                        candidate_tests = self.models[i].generate_test(self.N_candidate_tests)
  
                        # Pick only the valid tests.
                        valid_idx = [i for i in range(self.N_candidate_tests) if self.sut.validity(candidate_tests[i]) == 1]
                        candidate_tests = candidate_tests[valid_idx]
                        invalid += self.N_candidate_tests - len(valid_idx)
                        if candidate_tests.shape[0] == 0:
                          continue
  
                        # Estimate objective function values and add the tests
                        # to heap.
                        tests_predicted_objective = self.models[i].predict_objective(candidate_tests)
                        for j in range(tests_predicted_objective.shape[0]):
                            heapq.heappush(heap, (1 - tests_predicted_objective[j,0], entry_count, candidate_tests[j]))
                            entry_count += 1
  
                        break
  
                target_fitness *= self.fitness_coef
  
                # Check if the best predicted test is good enough.
                if 1 - heap[0][0] >= target_fitness: break
  
            # Save information on how many tests needed to be generated etc.
            # -----------------------------------------------------------------
            self.save_history("generation_time", self.timer_reset("generation"))
            N_generated = rounds*self.N_candidate_tests
            self.save_history("N_tests_generated", N_generated)
            self.save_history("N_invalid_tests_generated", invalid)
  
            # Execute the test on the SUT.
            # -----------------------------------------------------------------
            best_test = heap[0][2]
            best_estimated_objective = 1 - heap[0][0]
  
            self.log("Chose test {} with predicted maximum objective {}. Generated total {} tests of which {} were invalid.".format(best_test, best_estimated_objective, rounds, invalid))
            self.log("Executing the test...")
  
            self.timer_start("execution")
            sut_output = self.sut.execute_test(best_test)
            self.save_history("execution_time", self.timer_reset("execution"))
            # Check if we get a vector of floats or a 2-tuple of arrays
            # (signals).
            if np.isscalar(sut_output[0]):
                output = self.objective_func(sut_output)
            else:
                output = self.objective_func(*sut_output)
  
            self.log("The actual objective {} for the generated test.".format(output))
  
            # Add the new test to the test suite.
            # -----------------------------------------------------------------
            idx = self.test_repository.record(best_test, output)
            self.test_suite.append(idx)
            self.objective_selector.update(np.argmax(output))
            tests_generated += 1
  
            # Train the model.
            # -----------------------------------------------------------------
            # We train the models which were involved in the test generation.
            # We do not take the updated active model into account. We train
            # only if enough delay since the last training.
            self.timer_start("training")
            for i in active_models:
                if tests_generated - model_trained[i] >= self.train_delay:
                    self.log("Training the OGAN model {}...".format(i + 1))
                    dataX, dataY = self.test_repository.get(self.test_suite)
                    dataX = np.array(dataX)
                    dataY = np.array(dataY)
                    for epoch in range(self.models[i].train_settings["epochs"]):
                        self.models[i].train_with_batch(dataX,
                                                        dataY,
                                                        train_settings=self.models[i].train_settings,
                                                       )
            self.save_history("training_time", self.timer_reset("training"))
  
            self.timers_hold()
            yield idx
            self.timers_resume()
  
