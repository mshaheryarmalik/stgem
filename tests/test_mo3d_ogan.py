import math, os, unittest

from stgem.budget import Budget
from stgem.generator import STGEM, Search
from stgem.sut.python import PythonFunction
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorMAB
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS

def myfunction(input: [[-15, 15], [-15, 15], [-15, 15]]) -> [[0, 350], [0, 350], [0, 350]]:
    x1, x2, x3 = input[0], input[1], input[2]
    h1 = 305 - 100 * (math.sin(x1 / 3) + math.sin(x2 / 3) + math.sin(x3 / 3))
    h2 = 230 - 75 * (math.cos(x1 / 2.5 + 15) + math.cos(x2 / 2.5 + 15) + math.cos(x3 / 2.5 + 15))
    h3 = (x1 - 7) ** 2 + (x2 - 7) ** 2 + (x3 - 7) ** 2 - (
            math.cos((x1 - 7) / 2.75) + math.cos((x2 - 7) / 2.75) + math.cos((x3 - 7) / 2.75))

    return [h1, h2, h3]

class TestPython(unittest.TestCase):
    def test_ogan(self):
        mode = "stop_at_first_objective"

        generator = STGEM(
            description="mo3d/OGAN",
            sut=PythonFunction(function=myfunction),
            budget=Budget(),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            objective_selector=ObjectiveSelectorMAB(warm_up=41),
            steps=[
                Search(budget_threshold={"executions": 20},
                       mode=mode,
                       algorithm=Random(model_factory=(lambda: Uniform(parameters={"min_distance": 0.2})))),
                Search(budget_threshold={"executions": 40},
                       mode=mode,
                       algorithm=Random(model_factory=(lambda: LHS(parameters={"samples": 20})))),
                Search(budget_threshold={"executions": 45},
                       mode=mode,
                       algorithm=OGAN(model_factory=(lambda: OGAN_Model())))
            ]
        )

        r = generator.run()
        file_name = "mo3d_python_ogan_results.pickle"
        r.dump_to_file(file_name)
        os.remove(file_name)

if __name__ == "__main__":
    unittest.main()

