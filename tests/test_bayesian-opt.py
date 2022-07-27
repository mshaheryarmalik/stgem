import os, unittest, math

from stgem.sut.python import PythonFunction
from stgem.algorithm.bayesian.algorithm import BayesOptSampler
from verifai.features.features import Box
from stgem.generator import STGEM, Search
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorAll

def myfunction(input: [[-15, 15], [-15, 15], [-15, 15]]) -> [[0, 350], [0, 350], [0, 350]]:
    x1, x2, x3 = input[0], input[1], input[2]
    h1 = 305 - 100 * (math.sin(x1 / 3) + math.sin(x2 / 3) + math.sin(x3 / 3))
    h2 = 230 - 75 * (math.cos(x1 / 2.5 + 15) + math.cos(x2 / 2.5 + 15) + math.cos(x3 / 2.5 + 15))
    h3 = (x1 - 7) ** 2 + (x2 - 7) ** 2 + (x3 - 7) ** 2 - (
            math.cos((x1 - 7) / 2.75) + math.cos((x2 - 7) / 2.75) + math.cos((x3 - 7) / 2.75))

    return [h1, h2, h3]

# NOTE!: requires verifai and GPyOpt package to run
# Valid Domains from Verifai package are Box(), Array(), Struct()

class TestBayesOptSampler(unittest.TestCase):
    def test_python(self):
        mode = "stop_at_first_objective"

        generator = STGEM(
            description="bayesianOpt",
            sut=PythonFunction(function=myfunction),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            objective_selector=ObjectiveSelectorAll(),
            steps=[Search(budget_threshold={"executions": 5},
                          mode=mode,
                          algorithm=BayesOptSampler(domain=Box((-1,1),(-1,1),(-1,1)), init_num=3))
                  ]
            )
        r = generator.run()
        file_name = "bayesianOpt_results.pickle"
        r.dump_to_file(file_name)
        os.remove(file_name)

if __name__ == "__main__":
    unittest.main()
