import dill as pickle
import unittest
from stgem.generator import STGEM, Search, Load
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform
from stgem.objective import Minimize
from stgem.sut.mo3d import MO3D
from stgem.objective_selector import ObjectiveSelectorAll

class TestLoad(unittest.TestCase):
    def test_load(self):
        mode = "stop_at_first_objective"

        generator = STGEM(
            description="mo3d/OGAN",
            sut=MO3D(),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            objective_selector=ObjectiveSelectorAll(),

            steps=[Load(file_name = "test.pickle", load_range=15),
                   Search(budget_threshold={"executions": 20},
                            mode=mode,
                            algorithm=Random(model_factory=(lambda: Uniform())))
                   ]
        )
        sr = generator.run()
        with open("test.pickle", "wb") as f:
            pickle.dump(sr, f)
        print("Hej")
