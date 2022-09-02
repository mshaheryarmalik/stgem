import unittest

import numpy as np

from stgem.generator import STGEM, Search
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.ogan.model_keras import OGANK_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, Halton, LHS
from stgem.algorithm.wogan.algorithm import WOGAN
from stgem.algorithm.wogan.model import WOGAN_Model
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorAll
from stgem.sut.mo3d.sut import MO3D

class TestFalsifySTL(unittest.TestCase):

    def test_model_skeletons(self):
        # Use Search steps with all algorithms that use models and check for
        # each step that the skeletons can be resetuped and used for inference.

        generator = STGEM(
            description="test-model-skeletons",
            sut=MO3D(),
            objectives=[Minimize(selected=[0,1,2], scale=True)],
            objective_selector=ObjectiveSelectorAll(),
            steps=[
                Search(budget_threshold={"executions": 5},
                       algorithm=Random(model_factory=(lambda: Uniform())),
                       results_include_models=True),
                Search(budget_threshold={"executions": 10},
                       algorithm=Random(model_factory=(lambda: Halton())),
                       results_include_models=True),
                Search(budget_threshold={"executions": 15},
                       algorithm=Random(model_factory=(lambda: LHS(parameters={"samples": 5}))),
                       results_include_models=True),
                #Search(budget_threshold={"executions": 17},
                #       algorithm=OGAN(model_factory=(lambda: OGANK_Model())),
                #       results_include_models=True),
                Search(budget_threshold={"executions": 19},
                       algorithm=OGAN(model_factory=(lambda: OGAN_Model())),
                       results_include_models=True),
                #Search(budget_threshold={"executions": 21},
                #       algorithm=WOGAN(model_factory=(lambda: WOGAN_Model())),
                #       results_include_models=True)
            ]
        )

        r = generator.run()

        search_space = generator.search_space
        device = generator.device
        X, _, Y = r.test_repository.get()
        dataX = np.array([x.inputs for x in X])
        dataY = np.array(Y)

        x = np.array([0,0,0]).reshape(1, -1)

        classes = [Uniform, Halton, LHS, OGAN_Model]
        #classes = [Uniform, Halton, LHS, OGANK_Model, OGAN_Model, WOGAN_Model]
        #for step_idx in range(6):
        for step_idx in range(4):
            model_skeleton = r.step_results[step_idx].models[0][0]
            t = model_skeleton.generate_test()
            y = model_skeleton.predict_objective(x)

            model = classes[step_idx].setup_from_skeleton(model_skeleton, search_space, device)
            t = model.generate_test()
            y = model.predict_objective(x)
            if isinstance(model, WOGAN_Model):
                model.train_with_batch(dataX)
            else:
                model.train_with_batch(dataX, dataY)

if __name__ == "__main__":
    unittest.main()

