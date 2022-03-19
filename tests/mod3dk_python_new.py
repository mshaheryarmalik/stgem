from stgem.generator import STGEM, Search
from stgem.sut.python.sut import PythonFunction
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorMAB
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model_keras import OGANK_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform

import math


def myfunction(input: [[-15, 15], [-15, 15], [-15, 15]]) -> [[0, 350], [0, 350], [0, 350]]:
    x1, x2, x3 = input[0], input[1], input[2]
    h1 = 305 - 100 * (math.sin(x1 / 3) + math.sin(x2 / 3) + math.sin(x3 / 3))
    h2 = 230 - 75 * (math.cos(x1 / 2.5 + 15) + math.cos(x2 / 2.5 + 15) + math.cos(x3 / 2.5 + 15))
    h3 = (x1 - 7) ** 2 + (x2 - 7) ** 2 + (x3 - 7) ** 2 - (
            math.cos((x1 - 7) / 2.75) + math.cos((x2 - 7) / 2.75) + math.cos((x3 - 7) / 2.75))

    return [h1, h2, h3]


generator = STGEM(
    description="mo3d/OGAN",
    sut=PythonFunction(function=myfunction),
    objectives=[Minimize(selected=[0], scale=True),
                Minimize(selected=[1], scale=True),
                Minimize(selected=[2], scale=True)
                ],
    objective_selector=ObjectiveSelectorMAB(warm_up=30),
    steps=[
        Search(max_tests=20,
               algorithm=Random(model_factory=(lambda: Uniform()))),
        Search(max_tests=20,
               mode="stop_at_first_objective",
               algorithm=OGAN(model_factory=(lambda: OGANK_Model()))
        )
    ]
)

r = generator.run()
r.dump_to_file("mo3k_python_results.pickle")
