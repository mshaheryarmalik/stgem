import unittest

from stgem.budget import Budget
from stgem.generator import STGEM, Search
from stgem.experiment import Experiment
from stgem.sut.hyper import HyperParameter, Range
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
        # We change the fitness coefficient in the OGAN step.
        def f(generator, value):
            generator.steps[1].algorithm.fitness_coefficient = value
        sut_parameters = {"hyperparameters": [[f, Range(0, 1)]]}

        def sut_factory():
            return MO3D()

        def objective_factory():
            return [Minimize(selected=[0], scale=True),
                    Minimize(selected=[1], scale=True),
                    Minimize(selected=[2], scale=True),
                   ]

        def objective_selector_factory():
            return ObjectiveSelectorMAB(warm_up=60)

        def step_factory():
            return [Search(budget_threshold={"executions": 50},
                           algorithm=Random(model_factory=(lambda: Uniform()))),
                    Search(budget_threshold={"executions": 300},
                           algorithm=OGAN(model_factory=(lambda: OGAN_Model())))
                   ]

        def generator_factory(sut_factory, objective_factory, objective_selector_factory, step_factory):
            def f():
                return STGEM(description="",
                             sut=sut_factory(),
                             budget=Budget(),
                             objectives=objective_factory(),
                             objective_selector=objective_selector_factory(),
                             steps=step_factory())

            return f

        def get_seed_factory():
            def seed_generator():
                c = 0
                while True:
                    yield c
                    c += 1

            g = seed_generator()
            return lambda: next(g)

        def experiment_factory():
            return Experiment(5, generator_factory(sut_factory, objective_factory, objective_selector_factory, step_factory), get_seed_factory())

        mode = "falsification_rate"
        generator = STGEM(
                          description="Hyperparameter search",
                          sut=HyperParameter(mode, experiment_factory, sut_parameters),
                          budget=Budget(),
                          objectives=[Minimize(selected=[0], scale=False)],
                          objective_selector=ObjectiveSelectorAll(),
                          steps=[
                              Search(budget_threshold={"executions": 5},
                                     algorithm=Random(model_factory=(lambda: Uniform())))
                          ]
        )

        r = generator.run()

        X, Z, Y = generator.test_repository.get()
        print(X)
        print(Y)
        print("--------")

if __name__ == "__main__":
    unittest.main()

