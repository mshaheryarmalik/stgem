class Experiment:

    def __init__(self, N, stgem_factory, seed_factory, generator_callback=None, result_callback=None):
        self.N = N
        self.stgem_factory = stgem_factory
        self.seed_factory = seed_factory
        self.generator_callback = generator_callback
        self.result_callback = result_callback

    def run(self, silent=False):
        for _ in range(self.N):
            generator = self.stgem_factory()
            seed = self.seed_factory()
            generator.setup(seed=seed, silent=silent)

            if not self.generator_callback is None:
                self.generator_callback(generator)

            r = generator._run(silent=silent)

            if not self.result_callback is None:
                self.result_callback(r)
            
