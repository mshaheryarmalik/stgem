import unittest
from stgem.experiment import Experiment
from stgem.budget import Budget
from stgem.generator import STGEM, Search
from stgem.sut.python import PythonFunction
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorMAB
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import MinimumDistance
import random
import numpy as np

class TestMinDist(unittest.TestCase):
    def myfunction(self, input: [[0, 120]]) -> [[-200, 200]]:
        x = input
        return 100-x
    def factory(self):
            self.sut = PythonFunction(function = self.myfunction)
            mode = "stop_at_first_objective"
            generator = STGEM(
                description="simple unit test",
                sut = self.sut,
                budget = Budget(),
                objectives = [Minimize(selected=[0], scale=True)],
                objective_selector=ObjectiveSelectorMAB(warm_up=5),
                steps = [
                        Search(budget_threshold={"executions": 20},
                        mode = mode,
                        algorithm= Random(model_factory=(lambda: MinimumDistance()), parameters = {"mindist": 0.01})
                    )
                ]
            )
            return generator
    def seed(self):
            return random.randint(0, 1000)
    def test_minimum_distance(self): 
            generator = self.factory()
            r = generator.run(seed = self.seed())
            for i in r.step_results[0].test_repository._tests:
                for j in r.step_results[0].test_repository._tests:
                    if i != j:
                        assert np.linalg.norm(np.asarray(i)-np.asarray(j)) >= generator.steps[0].algorithm.models[0].mindist
if __name__ == "__main__":
    unittest.main()