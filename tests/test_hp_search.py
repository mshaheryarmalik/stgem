import unittest

import numpy as np

from stgem.generator import STGEM, Search
from stgem.experiment import Experiment
from stgem.sut.hyper import HyperParameter, Range, Categorical
from stgem.sut.mo3d import MO3D
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorAll, ObjectiveSelectorMAB
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model_keras import OGANK_Model
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS

class TestPython(unittest.TestCase):

    def test_hp_search(self):
        # We change the learning rates of the discriminator and the generator.
        def f1(generator, value):
            # Setup on generator has been called already, so the model objects
            # exist. We edit their parameter dictionaries and resetup them.
            for model in generator.steps[1].algorithm.models:
                model.parameters["discriminator_lr"] = value
                model.setup(model.search_space, model.device, model.logger)
        def f2(generator, value):
            # Similar to above.
            for model in generator.steps[1].algorithm.models:
                model.parameters["generator_lr"] = value
                model.setup(model.search_space, model.device, model.logger)

        hp_sut_parameters = {"hyperparameters": [[f1, Categorical([0.1, 0.01, 0.001, 0.0001])], [f2, Categorical([0.1, 0.01, 0.001, 0.0001])]],
                             "mode":            "falsification_rate"
                            }

        def sut_factory():
            return MO3D()

        def objective_factory():
            return [Minimize(selected=[0], scale=True),
                    Minimize(selected=[1], scale=True),
                    Minimize(selected=[2], scale=True),
                   ]

        def objective_selector_factory():
            return ObjectiveSelectorMAB(warm_up=25)

        def step_factory():
            return [Search(budget_threshold={"executions": 20},
                           algorithm=Random(model_factory=(lambda: Uniform()))),
                    Search(budget_threshold={"executions": 80},
                           algorithm=OGAN(model_factory=(lambda: OGAN_Model())))
                   ]

        def get_generator_factory(sut_factory, objective_factory, objective_selector_factory, step_factory):
            def generator_factory():
                return STGEM(description="",
                             sut=sut_factory(),
                             objectives=objective_factory(),
                             objective_selector=objective_selector_factory(),
                             steps=step_factory())

            return generator_factory

        def get_seed_factory():
            def seed_generator():
                c = 25321
                while True:
                    yield c
                    c += 1

            g = seed_generator()
            return lambda: next(g)

        def experiment_factory():
            N = 2
            return Experiment(N, get_generator_factory(sut_factory, objective_factory, objective_selector_factory, step_factory), get_seed_factory())

        # Note: Latin hypercube design makes sense with categorical values.
        # Otherwise certain values can be never considered.

        generator = STGEM(
                          description="Hyperparameter search",
                          sut=HyperParameter(experiment_factory, hp_sut_parameters),
                          objectives=[Minimize(selected=[0], scale=False)],
                          objective_selector=ObjectiveSelectorAll(),
                          steps=[
                              Search(budget_threshold={"executions": 2},
                                     algorithm=Random(model_factory=(lambda: LHS(parameters={"samples": 5}))))
                          ]
        )

        r = generator.run()

if __name__ == "__main__":
    unittest.main()

