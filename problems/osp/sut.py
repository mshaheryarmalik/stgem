import math
import numpy as np
import inspect
from stgem.sut import SUT, SUTOutput
from utils import *


class OSPSUT(SUT):
    """
    A SUT which encapsulates a OSP Scenario which we assume to take vectors
    as inputs and output vectors.
    """

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # TODO : Define correct current velocity variable ranges
        self.input_range = [[-15, 15], [-15, 15], [-15, 15], [-15, 15], [-15, 15], [-15, 15]]
        self.output_range = [[0, 350], [0, 350], [0, 350], [0, 350], [0, 350], [0, 350]]
        self.input_type = "vector"
        self.output_type = "vector"
        self.case_number = 0

    def _execute_test(self, test):
        # Get test case from SUTInput object and extract the required information
        current_velocity = self.descale(test.inputs.reshape(1, -1), self.input_range).reshape(-1)

        output = []
        error = None
        # Exception handler
        try:
            # Run OSP simulation
            sim_result = execute_test_case(self.case_number, current_velocity)
            # TODO: Process the results and extract current velocity as outputs
            output = sim_result
            self.case_number = self.case_number + 1
        except Exception as err:
            error = err

        test.input_denormalized = current_velocity

        return SUTOutput(np.asarray(output), None, None, error)
