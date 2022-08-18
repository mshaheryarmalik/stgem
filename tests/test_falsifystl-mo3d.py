import math
import unittest

import stl.robustness as STL

from stgem.generator import STGEM, Search
from stgem.sut.python import PythonFunction
from stgem.objective import FalsifySTL

from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model_keras import OGANK_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform

def myfunction(input: [[-15, 15], [-15, 15], [-15, 15]]) -> [[0, 350], [0, 350], [0, 350]]:
    x1, x2, x3 = input[0], input[1], input[2]
    h1 = 305 - 100 * (math.sin(x1 / 3) + math.sin(x2 / 3) + math.sin(x3 / 3))
    h2 = 230 - 75 * (math.cos(x1 / 2.5 + 15) + math.cos(x2 / 2.5 + 15) + math.cos(x3 / 2.5 + 15))
    h3 = (x1 - 7) ** 2 + (x2 - 7) ** 2 + (x3 - 7) ** 2 - (
            math.cos((x1 - 7) / 2.75) + math.cos((x2 - 7) / 2.75) + math.cos((x3 - 7) / 2.75))

    return [h1, h2, h3]

class TestFalsifySTL(unittest.TestCase):

    def test_python_multiple(self):
        from stgem.objective_selector.objective_selector import ObjectiveSelectorMAB

        F1 = "o0 > 0"
        F2 = "o1 > 0"
        F3 = "o2 > 0"
        ranges = {"o0": [0, 350], "o1": [0, 350], "o2": [0, 350]}

        generator = STGEM(
            description="mo3d-OGAN",
            sut=PythonFunction(function=myfunction),
            objectives=[FalsifySTL(specification=F1, ranges=ranges, scale=True),
                       FalsifySTL(specification=F2, ranges=ranges, scale=True),
                       FalsifySTL(specification=F3, ranges=ranges, scale=True)],
            objective_selector=ObjectiveSelectorMAB(),
            steps=[
                Search(budget_threshold={"executions": 20},
                       algorithm=Random(model_factory=(lambda: Uniform()))),
                Search(budget_threshold={"executions": 25},
                       mode="stop_at_first_objective",
                       algorithm=OGAN(model_factory=(lambda: OGANK_Model()))
                       )
            ]
        )

        r = generator.run()

    def test_MOD3D(self):
        from stgem.sut.mo3d.sut import MO3D

        specification = "o0 > 0 and o1 > 0 and o2 > 0"
        ranges = {"o0": [0, 350], "o1": [0, 350], "o2": [0, 350]}

        generator = STGEM(
            description="mo3d-OGAN",
            sut=MO3D(),
            objectives=[FalsifySTL(specification=specification, ranges=ranges, scale=True)],
            steps=[
                Search(budget_threshold={"executions": 20},
                       algorithm=Random(model_factory=(lambda: Uniform()))),
                Search(budget_threshold={"executions": 25},
                       mode="stop_at_first_objective",
                       algorithm=OGAN(model_factory=(lambda: OGANK_Model()))
                       )
            ]
        )

        r = generator.run()

if __name__ == "__main__":
    unittest.main()

