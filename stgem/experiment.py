import gc
from multiprocess import Process, Queue

# Notice that the callbacks need to understand that calls can arrive out of
# order.

class Experiment:

    def __init__(self, N, stgem_factory, seed_factory, generator_callback=None, result_callback=None):
        self.N = N
        self.stgem_factory = stgem_factory
        self.seed_factory = seed_factory
        self.generator_callback = generator_callback
        self.result_callback = result_callback

    def run(self, N_workers=1, silent=False):
        if N_workers < 1:
            raise SystemExit("The number of workers must be positive.")
        elif N_workers == 1:
            # Do not use multiprocessing.
            for _ in range(self.N):
                generator = self.stgem_factory()
                seed = self.seed_factory()
                generator.setup(seed=seed)

                if silent:
                    generator.logger.silent = True

                if not self.generator_callback is None:
                    self.generator_callback(generator)

                r = generator._run()

                if not self.result_callback is None:
                    self.result_callback(r)

                # Delete generator and force garbage collection. This is
                # especially important when using Matleb SUTs as several
                # Matlab instances take quite lot of memory.
                del generator
                gc.collect()
        else:
            # Use multiprocessing.
            def consumer(queue, silent, generator_callback, result_callback):
                while True:
                    msg = queue.get()
                    if msg == "STOP": break

                    generator, seed = msg

                    generator.setup(seed=seed)

                    if silent:
                        generator.logger.silent = True

                    if not generator_callback is None:
                        generator_callback(generator)

                    r = generator._run()

                    if not result_callback is None:
                        result_callback(r)

                    # Delete and garbage collect. See above.
                    del generator
                    gc.collect()
                    
            def producer(queue, N_workers, N, stgem_factory, seed_factory):
                for _ in range(N):
                    queue.put((stgem_factory(), seed_factory()))

                for _ in range(N_workers):
                    queue.put("STOP")

            queue = Queue(maxsize=N_workers)

            workers = []
            for _ in range(N_workers):
                consumer_process = Process(target=consumer, args=[queue, silent, self.generator_callback, self.result_callback], daemon=True)
                workers.append(consumer_process)
                consumer_process.start()

            producer(queue, N_workers, self.N, self.stgem_factory, self.seed_factory)

            for consumer_process in workers:
                consumer_process.join()

