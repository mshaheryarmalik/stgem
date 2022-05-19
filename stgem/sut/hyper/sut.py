import numpy as np

from stgem.sut import SUT, SUTOutput

class Range:
    """Continuous range [A, B]."""

    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __call__(self, x):
        # Scale from [-1, 1] to [A, B].
        return 0.5*((self.B - self.A)*x + self.A + self.B)

class Categorical:
    """Categorical variable."""

    def __init__(self, values):
        self.values = values

    def __call__(self, x):
        # We split [-1, 1] into len(self.values) parts, the first part
        # corresponds to the first variable value, second to second variable,
        # etc.
        idx = int(0.5*(x + 1)*len(self.values)) if x < 1 else len(self.values) - 1
        return self.values[idx]

class HyperParameter(SUT):

    def __init__(self, mode, experiment_factory, parameters=None):
        super().__init__(parameters)

        self.mode = mode
        self.experiment_factory = experiment_factory

        self.idim = len(self.hyperparameters)
        self.input_type = "vector"
        self.output_type = "vector"

        # Check what type of thing is computed: FR etc.
        if self.mode == "falsification_rate":
            self.stored = []

            def callback(result):
                """Append 1 if falsified, 0 otherwise."""

                self.stored.append(any(step.success for step in result.step_results))

            def report():
                # We return 1 - FR as we do minimization.
                return 1 - sum(1 if x else 0 for x in self.stored) / len(self.stored)

            self.stgem_result_callback = callback
            self.report = report

            self.odim = 1
        else:
            raise Exception("Unknown mode '{}'.".format(falsification_rate))

    def edit_generator(self, generator, test):
        denormalized = np.zeros_like(test.inputs)
        for n, (hp_func, hp_domain) in enumerate(self.hyperparameters):
            denormalized[n] = hp_domain(test.inputs[n])
            hp_func(generator, denormalized[n])

        test.input_denormalized = denormalized

    def _execute_test(self, test):
        experiment = self.experiment_factory()
        experiment.generator_callback = lambda g: self.edit_generator(g, test)
        experiment.result_callback = self.stgem_result_callback

        experiment.run(silent=True)

        return SUTOutput(np.array([self.report()]), None, None)

