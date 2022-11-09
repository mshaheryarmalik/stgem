import numpy as np
from stgem.sut import SUT, SUTOutput

class ModelBasedSUT(SUT):
    """A SUT which uses one or more models to produce its output.

    We expect that the models are already trained. This can be useful if we
    want to explore what a model has learned after testing a system."""

    def __init__(self, models, parameters=None):
        super().__init__(parameters)

        if len(models) == 0:
            raise ValueError("At least one model must be specified.")

        self.models = models

        self.idim = self.models[0].input_dimension
        self.input_type = "vector"
        # Check that the remaining models have compatible input dimension.
        for i in range(1, len(self.models)):
            if self.models[i].input_dimension != self.idim:
                raise ValueError("All models must have the same input dimension.")

        # The output dimension is the number of provided models.
        self.odim = len(self.models)
        self.output_type = "vector"
        # All models have the same input and output ranges.
        self.input_range  = [[-1, 1]] * self.idim
        self.output_range = [[0, 1]] * self.odim

    def _execute_test(self, test):
        output = np.zeros(self.odim)
        for i in range(self.odim):
            output[i] = self.models[i].predict_objective(test.inputs.reshape(1, -1))[0]

        test.input_denormalized = test.inputs

        return SUTOutput(output, None, None, None)

