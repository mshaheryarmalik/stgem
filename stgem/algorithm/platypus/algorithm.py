from stgem.algorithm import Algorithm

import numpy as np

from platypus import NSGAII, EpsMOEA, GDE3, SPEA2, Problem, Real, Integer, nondominated

class PlatypusOpt(Algorithm):
    """
    Implements the online generative adversarial network algorithm.
    """

    platypus_algorithm = {"NSGAII": NSGAII, "EpsMOEA": EpsMOEA, "GDE3": GDE3, "SPEA2": SPEA2}
    default_parameters = {"platypus_algorithm": "NSGAII"}

    def generate_test(self):
        self.perf.timer_start("generation")

        self.lastIdx=0
        self.reportedIdx=0

        # TODO: Add functionality related to invalid tests. Currently all tests
        #       are treated as valid.

        def fitness_func(test):
            # Save information on how many tests needed to be generated etc.
            # -----------------------------------------------------------------
            self.perf.save_history("generation_time", self.perf.timer_reset("generation"))
            self.perf.save_history("N_tests_generated", 1)
            self.perf.save_history("N_invalid_tests_generated", 0)

            # Execute the test on the SUT.
            # -----------------------------------------------------------------
            # Consume generation budget.
            self.budget.consume("generation_time", self.perf.get_history("generation_time")[-1])

            sut_output = self.sut.execute_test(np.array(test))
            self.log("Output from the SUT {}".format(sut_output))
            output = [self.objective_funcs[i](sut_output) for i in range(self.sut.odim)]

            self.log("The actual objective {} for the generated test.".format(output))

            # Add the new test to the test suite.
            # -----------------------------------------------------------------
            idx = self.test_repository.record(test, sut_output, output)
            self.lastIdx = idx
            self.objective_selector.update(np.argmin(output))

            self.perf.timer_start("generation")

            return output

        problem = Problem(self.sut.idim, self.sut.odim, 0)

        for i in range(self.sut.idim):
            problem.types[i] = Real(-1, 1)

        problem.function = fitness_func
        problem.directions[:] = Problem.MINIMIZE

        algorithm = self.platypus_algorithm[self.parameters.get("platypus_algorithm", "NSGAII")](
            problem,
            population_size=self.parameters.get("population_size", 100)
        )

        # It seems that NSGAII works in batches of more or less 100 evals
        # So each step generates many tests
        # We report them one by one. It is not the best solution, but it
        # kind of works for now.

        while True:
            algorithm.step()
            while self.reportedIdx <= self.lastIdx:
                yield self.reportedIdx
                self.reportedIdx += 1

