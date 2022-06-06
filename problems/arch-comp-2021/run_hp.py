import importlib, os, sys

import click
import numpy as np

from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS
from stgem.experiment import Experiment
from stgem.generator import STGEM, Search
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorAll
from stgem.sut.hyper import HyperParameter, Range, Categorical

from run import get_generator_factory, get_seed_factory, get_sut_objective_factory, benchmarks, specifications

@click.command()
@click.argument("selected_benchmark", type=click.Choice(benchmarks, case_sensitive=False))
@click.argument("selected_specification", type=str)
@click.argument("mode", type=str, default="")
@click.argument("init_seed_experiments", type=int)
@click.argument("seed_hp", type=int)
def main(selected_benchmark, selected_specification, mode, init_seed_experiments, seed_hp):
    if not selected_specification in specifications[selected_benchmark]:
        raise Exception("No specification '{}' for benchmark {}.".format(selected_specification, selected_benchmark))

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
                         "N_workers":       2}

    benchmark_module = importlib.import_module("{}.benchmark".format(selected_benchmark.lower()))

    def experiment_factory():
        N = 25
        sut_factory, objective_factory = get_sut_objective_factory(benchmark_module, selected_specification, mode)
        return Experiment(N, get_generator_factory("", sut_factory, objective_factory, benchmark_module.get_objective_selector_factory(), benchmark_module.get_step_factory()), get_seed_factory(init_seed_experiments))

    generator = STGEM(
                      description="Hyperparameter search for benchmark {} and specification {}".format(selected_benchmark, selected_specification),
                      sut=HyperParameter(experiment_factory, hp_sut_parameters),
                      objectives=[Minimize(selected=[0], scale=False)],
                      objective_selector=ObjectiveSelectorAll(),
                      steps=[
                          Search(budget_threshold={"executions": 64},
                                 algorithm=Random(model_factory=(lambda: LHS(parameters={"samples": 64}))))
                      ]
    )

    r = generator.run(seed=seed_hp)

    X, _, Y = generator.test_repository.get()
    Y = np.array(Y).reshape(-1)
    for n in range(len(X)):
        X2 = [hp_sut_parameters["hyperparameters"][i][1](x) for i, x in enumerate(X[n].inputs)]
        print("{} -> {}".format(X2, 1 - Y[n]))

if __name__ == "__main__":
    main()

