import os, unittest

from stgem.algorithm.random.model import BayesOptSampler
from stgem.generator import STGEM, Search
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorAll
from problems.odroid.sut import OdroidSUT
os.chdir("../problems/odroid")

class TestPython(unittest.TestCase):
    def test_odroid(self):
        mode = "stop_at_first_objective"

        generator = STGEM(
            description="mo3d-SA",
            sut=OdroidSUT(parameters = {"data_file": "odroid.npy"}),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            objective_selector=ObjectiveSelectorAll(),
            steps=[Search(budget_threshold={"executions": 10},
                          mode=mode,
                          algorithm=BayesOptSampler())
                  ]
            )
        r = generator.run()
        file_name = "mo3d_python_SA_results.pickle"
        r.dump_to_file(file_name)
        os.remove(file_name)

if __name__ == "__main__":
    unittest.main()
