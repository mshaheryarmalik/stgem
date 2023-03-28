import math
import numpy as np
import inspect
from stgem.sut import SUT, SUTOutput


class OSPSUT(SUT):
    """
    A SUT which encapsulates a OSP Scenario which we assume to take vectors
    as inputs and output vectors.
    """

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # TODO : Define WaveForm variables range
        self.input_range = [[-15, 15], [-15, 15], [-15, 15], [-15, 15], [-15, 15], [-15, 15]]
        self.output_range = [[0, 350], [0, 350], [0, 350], [0, 350], [0, 350], [0, 350]]
        self.input_type = "vector"
        self.output_type = "vector"

    def _execute_test(self, test):
        # TODO
        # Get test case from SUTInput object
        # Extract the required information
        # Prepare for OSP simulation
        # Run OSP simulation
        # Process the results and return results

        denormalized = self.descale(test.inputs.reshape(1, -1), self.input_range).reshape(-1)
        output = []
        error = None
        # Add a exception handler
        try:
            output = self.function(denormalized)
        except Exception as err:
            error = err

        test.input_denormalized = denormalized

        return SUTOutput(np.asarray(output), None, None, error)
