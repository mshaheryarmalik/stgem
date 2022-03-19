#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from stgem.sut import SUT


class PythonFunction(SUT):
    """
    A SUT which encapsulates a Python function.
    """

    def __init__(self, function, parameters=None):
        super().__init__(parameters)
        self.function = function

        # Use input parameters primarily and function annotation secondarily.
        if "input_range" in self.parameters and len(self.parameters["input_range"]) > 0:
            self.input_range = self.parameters["input_range"]
        else:
            for k, v in self.function.__annotations__.items():
                if k != "return":
                    self.input_range = v

        if "output_range" in self.parameters and len(self.parameters["output_range"]) > 0:
            self.output_range = self.parameters["output_range"]
        else:
            for k, v in self.function.__annotations__.items():
                if k == "return":
                    self.output_range = v

        self.idim = len(self.input_range)
        self.odim = len(self.output_range)

    def _execute_test(self, test):
        test = self.descale(test.reshape(1, -1), self.input_range).reshape(-1)
        output = self.function(test)

        return np.asarray(output)
