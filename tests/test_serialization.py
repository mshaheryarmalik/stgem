import os, math, unittest

from stgem.budget import Budget
from stgem.generator import STGEM, Search, STGEMResult
from stgem.sut.python.sut import PythonFunction
from stgem.objective import Minimize
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform

def myfunction(input: [[-15, 15], [-15, 15], [-15, 15]]) -> [[0, 350], [0, 350], [0, 350]]:
    x1, x2, x3 = input[0], input[1], input[2]
    h1 = 305 - 100 * (math.sin(x1 / 3) + math.sin(x2 / 3) + math.sin(x3 / 3))
    h2 = 230 - 75 * (math.cos(x1 / 2.5 + 15) + math.cos(x2 / 2.5 + 15) + math.cos(x3 / 2.5 + 15))
    h3 = (x1 - 7) ** 2 + (x2 - 7) ** 2 + (x3 - 7) ** 2 - (
            math.cos((x1 - 7) / 2.75) + math.cos((x2 - 7) / 2.75) + math.cos((x3 - 7) / 2.75))

    return [h1, h2, h3]

class MyTestCase(unittest.TestCase):
    def test_dump(self):
        mode = "stop_at_first_objective"

        generator = STGEM(
            description="test-dump",
            budget=Budget(),
            sut=PythonFunction(function=myfunction),
            objectives=[Minimize(selected=[0, 1, 2], scale=True)],
            steps=[
                Search(budget_threshold={"executions": 20},
                       mode=mode,
                       algorithm=Random(model_factory=(lambda: Uniform())))
            ]
        )

        r = generator.run()
        file_name = generator.description + ".pickle"
        r.dump_to_file(file_name)
        result2 = STGEMResult.restore_from_file(file_name)
        for s in result2.step_results:
            print(s.success)

        os.remove(file_name)

if __name__ == "__main__":
    unittest.main()
