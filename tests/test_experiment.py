import unittest
from stgem.experiment import Experiment
from stgem.budget import Budget
from stgem.generator import STGEM, Search
from stgem.sut.python import PythonFunction
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorMAB
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform
import random
import dill as pickle


class TestExperiment(unittest.TestCase):

    def myfunction(self, input: [[0, 120]]) -> [[-200, 200]]:
        x = input
        return 100 - x

    def factory(self):
        mode = "stop_at_first_objective"
        generator = STGEM(
            description="simple unit test",
            sut=PythonFunction(function=self.myfunction),
            budget=Budget(),
            objectives=[Minimize(selected=[0], scale=True)],
            objective_selector=ObjectiveSelectorMAB(warm_up=5),
            steps=[
                Search(budget_threshold={"executions": 20},
                       mode=mode,
                       algorithm=Random(model_factory=(lambda: Uniform()))
                       )
            ]
        )
        return generator

    def seed(self):
        return random.randint(0, 1000)

    def myCallback(self, result, generator, done):

        output = pickle.dumps(done)
        with open("done.pickle", "wb") as f:
            f.write(output)


    def test_experiment(self):
        print("running")
        experiment = Experiment(2, stgem_factory = self.factory, seed_factory = self.seed, result_callback = self.myCallback)
        experiment.run()
    #   experiment2 = Experiment(1, self.factory, self.seed, generator_callback = self.myCallback2)
    #  experiment2.run()


if __name__ == "__main__":
    unittest.main()