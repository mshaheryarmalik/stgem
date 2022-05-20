import importlib, os, sys

import numpy as np

from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.ogan.model_keras import OGANK_Model
from stgem.algorithm.wogan.algorithm import WOGAN
from stgem.algorithm.wogan.model import WOGAN_Model
from stgem.algorithm.random.model import Uniform, LHS
from stgem.budget import Budget
from stgem.experiment import Experiment
from stgem.generator import STGEM, Search
from stgem.objective import Minimize, FalsifySTL
from stgem.objective_selector import ObjectiveSelectorAll
from stgem.sut.hyper import HyperParameter, Range, Categorical

from util import build_specification

sys.path.append(os.path.join("problems", "arch-comp-2021"))
sys.path.append(os.path.join("problems", "arch-comp-2021", "f16"))
from common import get_generator_factory, get_seed_factory
from f16 import mode, ogan_parameters, ogan_model_parameters

if __name__ == "__main__":
    selected_specification = sys.argv[1]
    init_seed_experiments = int(sys.argv[2])
    seed_hp = int(sys.argv[3])

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

    hp_sut_parameters = {"hyperparameters": [[f1, Categorical([0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])],
                                             [f2, Categorical([0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])]],
                         "mode":            "falsification_rate",
                         "N_workers":       10}

    epsilon = 0.0
    sut, specifications, scale, strict_horizon_check = build_specification(selected_specification)

    def sut_factory():
        return sut

    def objective_factory():
        return [FalsifySTL(specification=specification, epsilon=epsilon, scale=scale, strict_horizon_check=strict_horizon_check) for specification in specifications]

    def objective_selector_factory():
        return ObjectiveSelectorAll()

    def step_factory():
        step_1 = Search(mode=mode,
                        budget_threshold={"executions": 75},
                        #algorithm=Random(model_factory=(lambda: LHS(parameters={"samples": 75})))
                        algorithm=Random(model_factory=(lambda: Uniform()))
                       )      
        step_2 = Search(mode=mode,
                        budget_threshold={"executions": 300},
                        #algorithm=WOGAN(model_factory=(lambda: WOGAN_Model()))
                        #algorithm=OGAN(model_factory=(lambda: OGANK_Model()))
                        algorithm=OGAN(model_factory=(lambda: OGAN_Model(ogan_model_parameters["dense"])), parameters=ogan_parameters)
                       )
        #steps = [step_1]
        steps = [step_1, step_2]
        return steps

    def experiment_factory():
        N = 25
        return Experiment(N, get_generator_factory(sut_factory, objective_factory, objective_selector_factory, step_factory), get_seed_factory(init_seed_experiments))

    generator = STGEM(
                      description="Hyperparameter search for F16",
                      sut=HyperParameter(experiment_factory, hp_sut_parameters),
                      budget=Budget(),
                      objectives=[Minimize(selected=[0], scale=False)],
                      objective_selector=ObjectiveSelectorAll(),
                      steps=[
                          Search(budget_threshold={"executions": 64},
                                 algorithm=Random(model_factory=(lambda: LHS(parameters={"samples": 5}))))
                      ]
    )

    r = generator.run(seed=seed_hp)

    X, _, Y = generator.test_repository.get()
    Y = np.array(Y).reshape(-1)
    for n in range(len(X)):
        X2 = [hp_sut_parameters["hyperparameters"][i][1](x) for i, x in enumerate(X[n].inputs)]
        print("{} -> {}".format(X2, 1 - Y[n]))

