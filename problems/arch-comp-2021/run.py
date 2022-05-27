import importlib, os, sys

import click

from stgem.budget import Budget
from stgem.generator import STGEM
from stgem.experiment import Experiment
from stgem.objective import FalsifySTL

def get_generator_factory(description, sut_factory, objective_factory, objective_selector_factory, step_factory):
    def generator_factory():
        return STGEM(description=description,
                     sut=sut_factory(),
                     budget=Budget(),
                     objectives=objective_factory(),
                     objective_selector=objective_selector_factory(),
                     steps=step_factory())

    return generator_factory

def get_seed_factory(init_seed=0):
    def seed_generator(init_seed):
        c = init_seed
        while True:
            yield c
            c += 1

    g = seed_generator(init_seed)
    return lambda: next(g)

def get_sut_objective_factory(benchmark_module, selected_specification, mode):
    sut, _, _, _, _ = benchmark_module.build_specification(selected_specification, mode)

    def sut_factory():
        return sut

    def objective_factory():
        _, specifications, scale, strict_horizon_check, epsilon = benchmark_module.build_specification(selected_specification, mode, sut)
        return [FalsifySTL(specification=specification, epsilon=epsilon, scale=scale, strict_horizon_check=strict_horizon_check) for specification in specifications]

    return sut_factory, objective_factory

benchmarks = ["AFC", "AT", "CC", "F16"]
descriptions = {
        "AFC": "Fuel Control of an Automotive Powertrain",
        "AT":  "Automatic Transmission",
        "CC":  "Chasing Cars",
        "F16": "Aircraft Ground Collision Avoidance System"
}
specifications = {
        "AFC": ["AFC27", "AFC29"],
        "AT":  ["AT1", "AT2", "AT51", "AT52", "AT53", "AT54", "AT6A", "AT6B", "AT6C", "AT6ABC", "ATX1", "ATX2"],
        "CC":  ["CC1", "CC2", "CC3", "CC4", "CC5", "CCX"],
        "F16": ["F16"]
}

@click.command()
@click.argument("selected_benchmark", type=click.Choice(benchmarks, case_sensitive=False))
@click.argument("selected_specification", type=str)
@click.argument("mode", type=str, default="")
@click.argument("n", type=int)
@click.argument("init_seed", type=int)
@click.argument("identifier", type=str, default="")
def main(selected_benchmark, selected_specification, mode, n, init_seed, identifier):
    N = n

    if not selected_specification in specifications[selected_benchmark]:
        raise Exception("No specification '{}' for benchmark {}.".format(selected_specification, selected_benchmark))

    def callback(result):
        path = os.path.join("..", "..", "output", selected_benchmark)
        file_name = "{}{}_{}.pickle".format(selected_specification, "_" + identifier if identifier is not None else "", str(result.timestamp).replace(" ", "_"))
        os.makedirs(path, exist_ok=True)
        result.dump_to_file(os.path.join(path, file_name))

    benchmark_module = importlib.import_module("{}.benchmark".format(selected_benchmark.lower()))

    sut_factory, objective_factory = get_sut_objective_factory(benchmark_module, selected_specification, mode)
    experiment = Experiment(N, get_generator_factory(descriptions[selected_benchmark], sut_factory, objective_factory, benchmark_module.get_objective_selector_factory(), benchmark_module.get_step_factory()), get_seed_factory(init_seed), result_callback=callback)

    N_workers = 2
    experiment.run(N_workers=N_workers, silent=False)

if __name__ == "__main__":
    main()

