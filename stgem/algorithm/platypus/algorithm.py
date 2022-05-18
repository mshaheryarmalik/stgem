from multiprocess import Process, JoinableQueue

import numpy as np
from platypus import NSGAII, EpsMOEA, GDE3, SPEA2, Problem, Real

from stgem.algorithm import Algorithm

class PlatypusOpt(Algorithm):

    platypus_algorithms = {"NSGAII": NSGAII, "EpsMOEA": EpsMOEA, "GDE3": GDE3, "SPEA2": SPEA2}
    default_parameters = {"platypus_algorithm": "NSGAII"}

    def initialize(self):
        self.queue = JoinableQueue()
        self.first_training = True

        problem = Problem(self.search_space.input_dimension, self.search_space.output_dimension, 0)

        for i in range(self.search_space.input_dimension):
            problem.types[i] = Real(-1, 1)

        problem.function = None
        problem.directions[:] = Problem.MINIMIZE

        self.algorithm = PlatypusOpt.platypus_algorithms[self.platypus_algorithm](
            problem,
            population_size=self.parameters.get("population_size", 100)
        )

    def _subprocess(self, queue, algorithm):
        def fitness_func(test):
            self.queue.put(test)
            self.queue.join()
            fitness = self.queue.get()

            return fitness

        algorithm.problem.function = fitness_func

        # It seems that NSGAII works in batches of more or less 100 evals, so
        # each step generates many tests. We report them one by one. It is not
        # the best solution, but it kind of works for now.

        while True:
            self.algorithm.step()

    def do_train(self, active_outputs, test_repository):
        if self.first_training:
            self.subprocess = Process(target=self._subprocess, args=[self.queue, self.algorithm], daemon=True)
            self.subprocess.start()
            self.first_training = False
        else:
            self.queue.put(test_repository.get(-1)[-1])
            self.queue.task_done()

    def do_generate_next_test(self, active_outputs, test_repository):
        test = self.queue.get()

        return np.array(test)

