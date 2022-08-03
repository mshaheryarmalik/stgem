import math, os, unittest

# Avoid a CUDA-related crash in the case that a CUDA device is available.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from stgem.experiment import Experiment
from stgem.generator import STGEM, Search
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorAll
from stgem.sut.mo3d import MO3D
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform

c = 0

class TestPython(unittest.TestCase):
    def test_ogan(self):
        mode = "stop_at_first_objective"

        def stgem_factory():
            generator = STGEM(
                description="mo3d-OGAN",
                sut=MO3D(),
                objectives=[Minimize(selected=[0], scale=True),
                            Minimize(selected=[1], scale=True),
                            Minimize(selected=[2], scale=True)
                            ],
                objective_selector=ObjectiveSelectorAll(),
                steps=[
                    Search(budget_threshold={"executions": 20},
                           mode=mode,
                           algorithm=Random(model_factory=(lambda: Uniform())))
                ]
            )

            return generator

        def get_seed_factory():
            def f():
                c = 0
                while True:
                    yield c
                    c += 1

            g = f()
            return lambda: next(g)

        def result_callback(idx, r, done):
            global c

            # Parallel execution may overwrite files here, but we do not care.

            k = c
            c += 1
            r.dump_to_file("mo3d_experiment_{}.pickle".format(k))

        N = 4
        experiment = Experiment(N, stgem_factory, get_seed_factory(), result_callback=result_callback)
        # The following works around a CI pipeline segfault.
        experiment.garbage_collect = False
        experiment.run(N_workers=2)

        for i in range(N):
            file_name = "mo3d_experiment_{}.pickle".format(i)
            try:
                os.remove(file_name)
            except:
                pass

if __name__ == "__main__":
    unittest.main()

