import heapq

import numpy as np

from stgem.algorithm import Algorithm

class OGAN(Algorithm):
    """
    Implements the online generative adversarial network algorithm.
    """

    # Do not change the defaults
    default_parameters = {"fitness_coef": 0.95, "train_delay": 1, "N_candidate_tests": 1, "reset_each_training": False}

    def setup(self, search_space, device=None, logger=None):
        super().setup(search_space, device, logger)
        self.first_training = True
        self.model_trained = [0 for m in range(self.N_models)] # keeps track how many tests were generated when a model was previously trained

    def do_train(self, active_outputs, test_repository, budget_remaining):
        # Take into account how many tests a previous step (usually random
        # search) has generated.
        self.tests_generated = test_repository.tests

        # TODO: Check that we have previously generated at least a couple of
        #       tests. Otherwise we get a cryptic error.

        # Train the models with the initial tests.
        # ---------------------------------------------------------------------
        # Notice that in principle we should train all models here. We however
        # opt to train only active models for more generality. It is up to the
        # caller to ensure that all models are trained here if so desired.

        for i in active_outputs:
            if self.first_training or self.tests_generated - self.model_trained[i] >= self.train_delay:
                self.log("Training the OGAN model {}...".format(i + 1))
                if not self.first_training and self.reset_each_training:
                    # Reset the model.
                    self.models[i].reset()
                X, _, Y = test_repository.get()
                dataX = np.asarray([sut_input.inputs for sut_input in X])
                dataY = np.array(Y)[:, i].reshape(-1, 1)
                epochs = self.models[i].train_settings["epochs"] if not self.first_training else self.models[i].train_settings_init["epochs"]
                for epoch in range(epochs):
                    if self.first_training:
                        train_settings = self.models[i].train_settings_init
                    else:
                        train_settings = self.models[i].train_settings
                    self.models[i].train_with_batch(dataX,
                                                    dataY,
                                                    train_settings=train_settings,
                                                    )
                self.model_trained[i] = self.tests_generated
        self.first_training = False

    def do_generate_next_test(self, active_outputs, test_repository, budget_remaining):
        heap = []
        target_fitness = 0
        entry_count = 0  # this is to avoid comparing tests when two tests added to the heap have the same predicted objective
        rounds = 0
        invalid = 0
        self.log("Generating using OGAN models {}.".format(",".join(str(m + 1) for m in active_outputs)))

        while True:
            # TODO: Avoid selecting similar or same tests.
            rounds += 1
            for i in active_outputs:
                while True:
                    # Generate several tests and pick the one with best
                    # predicted objective function component. We do this as
                    # long as we find at least one valid test.
                    try:
                        candidate_tests = self.models[i].generate_test(self.N_candidate_tests)
                    except:
                        raise

                    # Pick only the valid tests.
                    valid_idx = [i for i in range(self.N_candidate_tests) if self.search_space.is_valid(candidate_tests[i]) == 1]
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
            target_fitness = 1 - self.fitness_coef * (1 - target_fitness)

            # Check if the best predicted test is good enough.
            # Without eps we could get stuck if prediction is always 1.0.
            eps = 1e-4
            if heap[0][0] - eps <= target_fitness: break

        # Save information on how many tests needed to be generated etc.
        # -----------------------------------------------------------------
        N_generated = rounds*self.N_candidate_tests
        self.perf.save_history("N_tests_generated", N_generated)
        self.perf.save_history("N_invalid_tests_generated", invalid)

        best_test = heap[0][3]
        best_model = heap[0][2]
        best_estimated_objective = heap[0][0]

        self.log("Chose test {} with predicted minimum objective {} on OGAN model {}. Generated total {} tests of which {} were invalid.".format(best_test, best_estimated_objective, best_model + 1, rounds, invalid))

        return best_test

