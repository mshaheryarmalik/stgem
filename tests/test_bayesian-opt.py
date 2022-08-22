import os, unittest, math

from stgem.algorithm.bayesian.algorithm import BayesOptSampler
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform
from stgem.generator import STGEM, Search
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorAll
from stgem.sut.mo3d import MO3D

class TestBayesOptSampler(unittest.TestCase):
    def test_python(self):
        mode = "stop_at_first_objective"

        generator = STGEM(
            description="bayesianOpt",
            sut=MO3D(),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            objective_selector=ObjectiveSelectorAll(),
            steps=[
                Search(budget_threshold={"executions": 2},
                       mode=mode,
                       algorithm=Random(model_factory=(lambda: Uniform()))),
                Search(budget_threshold={"executions": 5},
                       mode=mode,
                       algorithm=BayesOptSampler())
                  ]
            )
        r = generator.run()

if __name__ == "__main__":
    unittest.main()

