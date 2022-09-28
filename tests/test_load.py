import os, unittest

from stgem.generator import STGEM, Search, Load
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform
from stgem.objective import Minimize
from stgem.sut.mo3d import MO3D
from stgem.objective_selector import ObjectiveSelectorAll

class TestLoad(unittest.TestCase):
    def test_load(self):
        mode_search = "stop_at_first_objective"

        # Create a pickle file to load from
        generator = STGEM(
            description="mo3d-OGAN",
            sut=MO3D(),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            objective_selector=ObjectiveSelectorAll(),

            steps=[Search(budget_threshold={"executions": 20},
                          mode=mode_search,
                          algorithm=Random(model_factory=(lambda: Uniform())))
                   ]
        )

        file_name = "test-load.pickle"
        r = generator.run()
        r.dump_to_file(file_name)

        #mode_load = "random"
        mode_load = "initial"

        generator = STGEM(
            description="mo3d-OGAN",
            sut=MO3D(),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            objective_selector=ObjectiveSelectorAll(),
            steps=[Load(file_name=file_name,
                        mode=mode_load,
                        range_load=15)
                  ]
        )

        r = generator.run()

        os.remove(file_name)

if __name__ == "__main__":
    unittest.main()

