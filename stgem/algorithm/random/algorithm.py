import numpy as np

from stgem.algorithm import Algorithm

class Random(Algorithm):
    """Baseline random algorithm for generating a test suite."""

    def do_train(self, active_outputs, test_repository, budget_remaining):
        pass

    def do_generate_next_test(self, active_outputs, test_repository, budget_remaining):
        # PerformanceRecordHandler for the current test.
        performance = test_repository.performance(test_repository.current_test)

        rounds = 0
        invalid = 0
        # Select a model randomly and generate a random valid test for it.
        m = np.random.choice(active_outputs)

        while True:
            rounds += 1
            new_test = self.models[m].generate_test().reshape(-1)
            if self.search_space.is_valid(new_test) == 0:
                invalid += 1
                continue

            break

        # Save information on how many tests needed to be generated etc.
        # -----------------------------------------------------------------
        performance.record("N_tests_generated", rounds)
        performance.record("N_invalid_tests_generated", invalid)

        return new_test

