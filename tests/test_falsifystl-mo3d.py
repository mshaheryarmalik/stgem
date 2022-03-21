from stgem.generator import STGEM, Search
from stgem.sut.python.sut import PythonFunction
from stgem.objective import FalsifySTL

from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model_keras import OGANK_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS

import math


def myfunction(input: [[-15, 15], [-15, 15], [-15, 15]]) -> [[0, 350], [0, 350], [0, 350]]:
    x1, x2, x3 = input[0], input[1], input[2]
    h1 = 305 - 100 * (math.sin(x1 / 3) + math.sin(x2 / 3) + math.sin(x3 / 3))
    h2 = 230 - 75 * (math.cos(x1 / 2.5 + 15) + math.cos(x2 / 2.5 + 15) + math.cos(x3 / 2.5 + 15))
    h3 = (x1 - 7) ** 2 + (x2 - 7) ** 2 + (x3 - 7) ** 2 - (
            math.cos((x1 - 7) / 2.75) + math.cos((x2 - 7) / 2.75) + math.cos((x3 - 7) / 2.75))

    return [h1, h2, h3]

import unittest

class TestFalsifySTL(unittest.TestCase):
    def test_python(self):
        generator = STGEM(
            description="mo3d/OGAN",
            sut=PythonFunction(function=myfunction),
            objectives=[FalsifySTL(specification="always[0,1] o0>0 and o1>0 and o2>0") ],
            steps=[
                Search(max_tests=20,
                       algorithm=Random(model_factory=(lambda: Uniform()))),
                Search(max_tests=5,
                       mode="stop_at_first_objective",
                       algorithm=OGAN(model_factory=(lambda: OGANK_Model()))
                )
            ]
        )

        r = generator.run()

    def test_python_multiple(self):
        from stgem.objective_selector.objective_selector import ObjectiveSelectorMAB
        generator = STGEM(
            description="mo3d/OGAN",
            sut=PythonFunction(function=myfunction),
            objectives=[FalsifySTL(specification="always[0,1] o0>0"),
                       FalsifySTL(specification="always[0,1] o1>0"),
                       FalsifySTL(specification="always[0,1] o2>0") ],
            objective_selector=ObjectiveSelectorMAB(),
            steps=[
                Search(max_tests=20,
                       algorithm=Random(model_factory=(lambda: Uniform()))),
                Search(max_tests=5,
                       mode="stop_at_first_objective",
                       algorithm=OGAN(model_factory=(lambda: OGANK_Model()))
                       )
            ]
        )

        r = generator.run()

    def test_MOD3D(self):
        from stgem.sut.mo3d.sut import MO3D
        generator = STGEM(
            description="mo3d/OGAN",
            sut=MO3D(),
            objectives=[FalsifySTL(specification="always[0,1] o0>0 and o1>0 and o2>0")],
            steps=[
                Search(max_tests=20,
                       algorithm=Random(model_factory=(lambda: Uniform()))),
                Search(max_tests=5,
                       mode="stop_at_first_objective",
                       algorithm=OGAN(model_factory=(lambda: OGANK_Model()))
                       )
            ]
        )

        r = generator.run()


if __name__ == "__main__":
    unittest.main()