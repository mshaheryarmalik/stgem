import unittest

from stgem.generator import STGEM, Search
from stgem.sut.matlab import Matlab
from stgem.objective import Minimize
from stgem.objective_selector.objective_selector import ObjectiveSelectorMAB

from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform

sut_parameters = {
    "model_file": "../problems/matlab/mo3d",
    "input_type": "vector",
    "output_type": "vector",
    "input_range": [[-15, 15], [-15, 15], [-15, 15]],
    "output_range":  [[0, 350], [0, 350], [0, 350]]
}
mode = "stop_at_first_objective"

class TestPython(unittest.TestCase):
    def test_python(self):
        generator = STGEM(
            description="Matlab-MO3D/OGAN",
            sut=Matlab(sut_parameters),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            objective_selector=ObjectiveSelectorMAB(warm_up=30),
            steps=[
                Search(budget_threshold={"executions": 20},
                       mode=mode,
                       algorithm=Random(model_factory=(lambda: Uniform()))),
                Search(budget_threshold={"executions": 80},
                       mode=mode,
                       algorithm=OGAN(model_factory=(lambda: OGAN_Model()))
                )
            ]
        )

        r = generator.run()
        r.dump_to_file("mo3k_python_results.pickle")

if __name__ == "__main__":
    unittest.main()

