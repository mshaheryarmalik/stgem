import math, os, unittest
from stgem.generator import STGEM, Search
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorMAB
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.ogan.model_keras import OGANK_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform
from sut import OSPSUT


class TestOSP(unittest.TestCase):
    def test_models(self):
        generator = STGEM(
            description="test_models",
            sut=OSPSUT(),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True),
                        Minimize(selected=[3], scale=True),
                        Minimize(selected=[4], scale=True),
                        Minimize(selected=[5], scale=True)
                        ],
            objective_selector=ObjectiveSelectorMAB(warm_up=10),
            steps=[
                Search(budget_threshold={"executions": 2},
                       algorithm=Random(model=Uniform(parameters={"min_distance": 0.2}))),
                Search(budget_threshold={"executions": 2},
                       algorithm=OGAN(model=OGAN_Model()))
            ]
        )
        r = generator.run()
        file_name = "osp_results1.pickle"
        r.dump_to_file(file_name)
        os.remove(file_name)

    def test_models_update(self):
        generator = STGEM(
            description="test_models",
            sut=OSPSUT(),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True),
                        Minimize(selected=[3], scale=True),
                        Minimize(selected=[4], scale=True),
                        Minimize(selected=[5], scale=True)
                        ],
            objective_selector=ObjectiveSelectorMAB(warm_up=10),
            steps=[
                Search(budget_threshold={"executions": 2},
                       algorithm=Random(model_factory=(lambda: Uniform(parameters={"min_distance": 0.2})))),
                Search(budget_threshold={"executions": 2},
                       algorithm=OGAN(models=[OGAN_Model(), OGANK_Model(), OGAN_Model]))
            ]
        )
        r = generator.run()
        file_name = "osp_results2.pickle"
        r.dump_to_file(file_name)
        os.remove(file_name)

    def test_factory(self):
        generator = STGEM(
            description="test_factory",
            sut=OSPSUT(),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True),
                        Minimize(selected=[3], scale=True),
                        Minimize(selected=[4], scale=True),
                        Minimize(selected=[5], scale=True)
                        ],
            objective_selector=ObjectiveSelectorMAB(warm_up=10),
            steps=[
                Search(budget_threshold={"executions": 2},
                       algorithm=Random(model_factory=(lambda: Uniform(parameters={"min_distance": 0.2})))),
                Search(budget_threshold={"executions": 2},
                       algorithm=OGAN(model_factory=(lambda: OGAN_Model())))
            ]
        )

        r = generator.run()
        file_name = "osp_results3.pickle"
        r.dump_to_file(file_name)
        os.remove(file_name)
