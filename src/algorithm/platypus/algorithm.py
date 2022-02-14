from algorithm.algorithm import Algorithm
import numpy as np

from platypus import NSGAII,EpsMOEA, Problem, Real, Integer, nondominated


class PlatypusOpt(Algorithm):
    """
    Implements the online generative adversarial network algorithm.
    """

    def __init__(self, sut, test_repository, objective_funcs, objective_selector, parameters, logger=None):
        super().__init__(sut, test_repository, objective_funcs, objective_selector, logger)
        self.parameters = parameters

    def generate_test(self):
        self.lastIdx=0
        self.reportedIdx=0

        def fitness_func(test):
            sut_output = self.sut.execute_test(np.array(test))

            # Check if the SUT output is a vector or a signal.
            if np.isscalar(sut_output[0]):
                output = [self.objective_funcs[i](sut_output) for i in range(self.sut.odim)]
            else:
                output = [self.objective_funcs[i](**sut_output) for i in range(self.sut.odim)]

            self.log("Result from the SUT {}".format(sut_output))
            self.log("The actual objective {} for the generated test.".format(output))

            # Add the new test to the test suite.
            # -----------------------------------------------------------------
            idx = self.test_repository.record(test, output)
            self.lastIdx=idx
            self.test_suite.append(idx)
            self.objective_selector.update(np.argmin(output))

            return output

        problem = Problem(self.sut.idim, self.sut.odim, 0)

        for i in range(self.sut.idim):
            problem.types[i] = Real(-1, 1)

        problem.function = fitness_func
        problem.directions[:] = Problem.MINIMIZE

        algorithm = NSGAII(problem)

        # It seems that NSGAII works in batches of more or less 100 evals
        # So each step generates many tests
        # We report them one by one. It is not the best solution, but it
        # kind of works for now.
        while True:
            algorithm.step()
            while self.reportedIdx<=self.lastIdx:
                yield self.reportedIdx
                self.reportedIdx=self.reportedIdx+1

