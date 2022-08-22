import os, sys, unittest

sys.path.append(os.path.join("..", ".."))

from stgem.generator import STGEM, Search
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorAll

from problems.odroid.sut import OdroidSUT

class TestPython(unittest.TestCase):
    def test_odroid(self):
        mode = "stop_at_first_objective"

        generator = STGEM(
            description="odroid",
            sut=OdroidSUT(parameters = {"data_file": "odroid.npy"}),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            objective_selector=ObjectiveSelectorAll(),
            steps=[Search(budget_threshold={"executions": 20},
                   mode=mode,
                   algorithm=Random(model_factory=(lambda: Uniform())))
                  ]
            )

        r = generator.run()

if __name__ == "__main__":
    unittest.main()

