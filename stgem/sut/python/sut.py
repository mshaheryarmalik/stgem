#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from stgem.sut import SUT

class PythonFunction(SUT):
    """
    A SUT which encapsulates a Python function.
    """

    def __init__(self, parameters):
        super().__init__(parameters)

        self.function = self.parameters["function"]

        # Use input parameters primarily and function annotation secondarily.
        if "input_range" in self.parameters and len(self.parameters["input_range"]) > 0:
            self.irange = self.parameters["input_range"]
        else:
            for k, v in self.function.__annotations__.items():
                if k != "return":
                    self.irange = v

        if "output_range" in self.parameters and len(self.parameters["output_range"]) > 0:
            self.orange = self.parameters["output_range"]
        else:
            for k, v in self.function.__annotations__.items():
                if k == "return":
                    self.orange = v

        self.idim = len(self.irange)
        self.odim = len(self.orange)

    def _execute_test(self, test):
        test = self.descale(test.reshape(1, -1), self.irange).reshape(-1)
        output = self.function(test)

        return np.asarray(output)

