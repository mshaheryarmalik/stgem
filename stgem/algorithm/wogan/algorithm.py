#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib, heapq

import numpy as np

from stgem.algorithm import Algorithm

class WOGAN(Algorithm):
    """
    Implements the test suite generation based on online Wasserstein generative
    adversarial networks.
    """

    def __init__(self, sut, test_repository, objective_funcs, objective_selector, parameters, logger=None):
        super().__init__(sut, test_repository, objective_funcs, objective_selector, parameters, logger)

        self.N_models = sum(1 for f in self.objective_funcs)

        # Setup the models.
        # ---------------------------------------------------------------------
        # Load the specified WOGAN model and initialize it.
        module = importlib.import_module("stgem.algorithm.wogan.model")
        self.model_class = getattr(module, self.wogan_model)
        self.models = [self.model_class(sut=self.sut, parameters=self.parameters, logger=logger) for _ in range(self.N_models)]

        # Setup the shift function for sampling training data.
        # ---------------------------------------------------------------------
        if self.shift_function is None:
            raise Exception("No shift function defined.")
        if self.shift_function_parameters is None:
            raise Exception("No shift function parameters defined.")

        if self.shift_function == "linear":
            # We increase the shift linearly according to the given initial and given
            # final value.
            M1 = self.max_tests + self.preceding_tests # total number of tests after WOGAN has completed
            M2 = self.preceding_tests                  # total number of tests before WOGAN started
            alpha = (self.shift_function_parameters["final"] - self.shift_function_parameters["initial"])/(M1 - M2)
            beta = self.shift_function_parameters["final"] - alpha * self.max_tests
            self.shift = lambda x: alpha * x + beta
        else:
            raise Exception("No shift function type '{}'.".format(self.shift_function))

        # Setup the function for computing the bin weights.
        # ---------------------------------------------------------------------
        self.bin_weight = lambda x: 1 / (1 + np.exp(-1 * x))

        self.get_bin = (lambda x: int(x * self.bins) if x < 1.0 else self.bins - 1)

    def bin_sample(self, N, shift):
        """
        Samples N bin indices.
        """

        """
        The distribution on the indices is defined as follows. Suppose that S
        is a nonnegative function satisfying S(-x) = 1 - x for all x. Consider
        the middle points of the bins. We map the middle point of the middle
        bin to 0 and the remaining middle points symmetrically around 0 with
        first middle point corresponding to -1 and the final to 1. We then
        shift these mapped middle points to the right by the given amount. The
        weight of the bin will is S(x) where x is the mapped and shifted middle
        point. We use the function self.bin_weight for S.
        """

        # If the number of bins is odd, then the middle point of the middle bin
        # interval is mapped to 0 and otherwise the point common to the two
        # middle bin intervals is mapped to 0.
        # TODO: This could be predefined in __init__.
        if self.bins % 2 == 0:
            h = lambda x: x - (int(self.bins/2) + 0.0) * (1/self.bins)
        else:
            h = lambda x: x - (int(self.bins/2) + 0.5) * (1/self.bins)

        # We basically take the middle point of a bin interval, map it to
        # [-1, 1] and apply S on the resulting point to find the unnormalized
        # bin weight.
        weights = np.zeros(shape=(self.bins))
        for n in range(self.bins):
            weights[n] = self.bin_weight(h((n + 0.5) * (1/self.bins)) - shift)
        # Normalize weights.
        weights = weights / np.sum(weights)

        idx = np.random.choice(list(range(self.bins)), N, p=weights)
        return idx

    def training_sample(self, N, X, B, shift):
        """
        Samples N elements from X. The sampling is done by picking a bin and
        uniformly randomly selecting a test from the bin, but we do not select
        the same test twice. The probability of picking each bin is computed
        via the function bin_sample.
        """

        sample_X = np.zeros(shape=(N, X.shape[1]))
        available = {n: v.copy() for n, v in B.items()}
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
        self.perf.timer_start("total")

        test_bins = [{i:[] for i in range(self.bins)} for _ in range(self.N_models)] # a dictionary to tell which test is in which bin for each model
        model_trained = [0 for _ in range(self.N_models)]                            # keeps track how many tests were generated when a model was previously trained
        tests_generated = 0                                                          # how many tests have been generated so far

        # Take into account how many tests a previous step (usually random
        # search) has generated.
        tests_generated = self.test_repository.tests

        # TODO: Check that we have previously generated at least a couple of
        #       tests. Otherwise we get a cryptic error.

        # Assign the initial tests to bins.
        for i in range(self.N_models):
            for j in self.test_repository.indices:
                test_bins[i][self.get_bin(self.test_repository.get(j)[1][i])].append(j)

        # Train the models with the initial tests.
        # ---------------------------------------------------------------------
        # Notice that in principle we should train all models here. We however
        # opt to train only active models for more generality. It is up to the
        # caller to ensure that all models are trained here if so desired.
        self.perf.timer_start("training")
        for i in self.objective_selector.select_all():
            dataX, dataY = self.test_repository.get()
            dataX = np.array(dataX)
            dataY = np.array(dataY)[:,i].reshape(-1, 1)
            for _ in range(self.models[i].train_settings["epochs"]):
                self.log("Training analyzer {}...".format(i + 1))
                # Train the analyzer.
                self.models[i].train_analyzer_with_batch(dataX,
                                                         dataY,
                                                         train_settings=self.models[i].train_settings_init,
                                                        )
                # Train the WGAN.
                self.log("Training WGAN model {}...".format(i + 1))
                for epoch in range(self.models[i].train_settings_init["epochs"]):
                    train_X = self.training_sample(min(self.wgan_batch_size, self.preceding_tests),
                                                   np.asarray(self.test_repository.get()[0]),
                                                   test_bins[i],
                                                   self.shift(tests_generated),
                                                  )
                    self.models[i].train_with_batch(train_X,
                                                    train_settings=self.models[i].train_settings_init,
                                                   )
            model_trained[i] = tests_generated
        self.perf.save_history("training_time", self.perf.timer_reset("training"))

        # Begin the main loop for new test generation and training.
        # ---------------------------------------------------------------------
        while True:
            # We generate a new valid test as follows. For each active model,
            # we generate new tests using the model, discard invalid tests, and
            # estimate the corresponding objective function values. The test
            # closest to target fitness 0 (minimization) is treated as the
            # candidate, and the candidate is accepted if the estimate exceeds
            # the current target threshold. The target threshold is changed on
            # each loop execution in order to make acceptance easier. We use a
            # priority queue to track the best tests in case that an estimated
            # good test was generated just before the target threshold was
            # changed enough for it to be selected.
            self.perf.timer_start("generation")
            heap = []
            target_fitness = 0
            entry_count = 0 # this is to avoid comparing tests when two tests added to the heap have the same predicted objective
            rounds = 0
            invalid = 0
            active_models = self.objective_selector.select()
            self.log("Starting to generate test {}.".format(tests_generated + 1))

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
                            heapq.heappush(heap, (tests_predicted_objective[j,0], entry_count, i, candidate_tests[j]))
                            entry_count += 1

                        break

                # We go up from 0 like we would go down from 1 when multiplied by self.fitness_coef.
                target_fitness = 1 - self.fitness_coef*(1 - target_fitness)

                # Check if the best predicted test is good enough.
                if heap[0][0] <= target_fitness: break

            # Save information on how many tests needed to be generated etc.
            # -----------------------------------------------------------------
            self.perf.save_history("generation_time", self.perf.timer_reset("generation"))
            N_generated = rounds*self.N_candidate_tests
            self.perf.save_history("N_tests_generated", N_generated)
            self.perf.save_history("N_invalid_tests_generated", invalid)

            # Execute the test on the SUT.
            # -----------------------------------------------------------------
            best_test = heap[0][3]
            best_model = heap[0][2]
            best_estimated_objective = heap[0][0]

            self.log("Chose test {} with predicted minimum objective {} on WGAN model {}. Generated total {} tests of which {} were invalid.".format(best_test, best_estimated_objective, best_model + 1, rounds, invalid))
            self.log("Executing the test...")

            sut_output = self.sut.execute_test(best_test)

            # Check if the SUT output is a vector or a signal.
            if np.isscalar(sut_output[0]):
                output = [self.objective_funcs[i](sut_output) for i in range(self.N_models)]
            else:
                output = [self.objective_funcs[i](*sut_output) for i in range(self.N_models)]
  
            self.log("Result from the SUT {}".format(sut_output))
            self.log("The actual objective {} for the generated test.".format(output))

            # Add the new test to the test suite.
            # -----------------------------------------------------------------
            idx = self.test_repository.record(best_test, output)
            self.objective_selector.update(np.argmin(output))
            tests_generated += 1

            # Place the test into appropriate bin for each model.
            for i in range(self.N_models):
                test_bins[i][self.get_bin(output[i])].append(idx)

            self.perf.timers_hold()
            yield idx
            self.perf.timers_resume()

            # Train the models.
            # -----------------------------------------------------------------
            # We train the models which were involved in the test generation.
            # We do not take the updated active model into account. We train
            # only if enough delay since the last training.
            self.perf.timer_start("training")
            for i in active_models:
                if tests_generated - model_trained[i] >= self.train_delay:
                    self.log("Training analyzer {}...".format(i + 1))
                    dataX, dataY = self.test_repository.get()
                    dataX = np.array(dataX)
                    dataY = np.array(dataY)[:,i].reshape(-1, 1)
                    for _ in range(self.models[i].train_settings_init["epochs"]):
                        self.models[i].train_analyzer_with_batch(dataX,
                                                                 dataY,
                                                                 train_settings=self.models[i].train_settings,
                                                                )

                        # Train the WGAN.
                        self.log("Training the WGAN model {}...".format(i + 1))
                        for epoch in range(self.models[i].train_settings_init["epochs"]):
                            # We include the new tests to the batch with high
                            # probability if and only if they have low objective.
                            BS = min(self.wgan_batch_size, self.preceding_tests)
                            train_X = np.zeros(shape=(BS, self.sut.idim))
                            c = 0
                            for j in range(self.train_delay):
                                test, output = self.test_repository.get(self.test_repository.indices[-(j+1)])
                                if self.get_bin(output[i]) >= self.bin_sample(1, self.shift(tests_generated))[0]:
                                    train_X[c] = test
                                    c += 1
                            train_X[c:] = self.training_sample(BS - c,
                                                               np.asarray(self.test_repository.get()[0]),
                                                               test_bins[i],
                                                               self.shift(tests_generated),
                                                              )
                            self.models[i].train_with_batch(train_X,
                                                            train_settings=self.models[i].train_settings,
                                                           )
            self.perf.save_history("training_time", self.perf.timer_reset("training"))

