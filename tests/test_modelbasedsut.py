import math, os, unittest

from stgem.generator import STGEM, Search
from stgem.sut.python import PythonFunction
from stgem.objective import Minimize
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model_keras import OGANK_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS
from stgem.sut.model import ModelBasedSUT

def myfunction(input: [[-15, 15], [-15, 15], [-15, 15]]) -> [[0, 350], [0, 350], [0, 350]]:
    x1, x2, x3 = input[0], input[1], input[2]
    h1 = 305 - 100 * (math.sin(x1 / 3) + math.sin(x2 / 3) + math.sin(x3 / 3))
    h2 = 230 - 75 * (math.cos(x1 / 2.5 + 15) + math.cos(x2 / 2.5 + 15) + math.cos(x3 / 2.5 + 15))
    h3 = (x1 - 7) ** 2 + (x2 - 7) ** 2 + (x3 - 7) ** 2 - (
            math.cos((x1 - 7) / 2.75) + math.cos((x2 - 7) / 2.75) + math.cos((x3 - 7) / 2.75))

    return [h1, h2, h3]

class TestModelBasedSUT(unittest.TestCase):

    def test_MBSUT(self):

        models = [OGANK_Model(), OGANK_Model(), OGANK_Model()]

        generator1 = STGEM(
            description="mo3d/mbst-actual",
            sut=PythonFunction(function=myfunction),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            steps=[
                Search(budget_threshold={"executions": 2},
                       algorithm=Random(model=Uniform(parameters={"min_distance": 0.2}))),
                Search(budget_threshold={"executions": 5},
                       algorithm=OGAN(models=models))
            ]
        )

        r = generator1.run()
        r.dump_to_file("tmp_mbsut_actual.pickle")

        # save models
        for (i, m) in zip(range(len(models)),models):
            m.save_to_file("tmp_mbsut_"+str(i))

        # restore models
        restored_models=[]

        for i in range(len(models)):
            m= OGANK_Model().load_from_file("tmp_mbsut_" + str(i))
            restored_models.append(m)

        generator2 = STGEM(
            description="mo3d/mbst-model",
            sut=ModelBasedSUT(models=restored_models),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            steps=[
                Search(budget_threshold={"executions": 20},
                       algorithm=Random(model=Uniform(parameters={"min_distance": 0.2})))
            ]
        )

        r2 = generator2.run()
        r2.dump_to_file("tmp_mbsut_model.pickle")

if __name__ == "__main__":
    unittest.main()

