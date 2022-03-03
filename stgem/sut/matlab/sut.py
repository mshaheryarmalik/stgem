#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys

import matlab.engine
import numpy as np
from stgem.sut import SUT
import os

# start the matlab engine
matlab_engine = matlab.engine.start_matlab()


class MATLAB(SUT):
    """
    Implements a SUT in matlab.
    """

    def __init__(self, parameters):
        SUT.__init__(self, parameters)

        self.parameters = parameters

        self.idim = len(self.parameters["input_range"])
        self.odim = len(self.parameters["output_range"])
        self.irange = np.asarray(self.parameters["input_range"])
        self.orange = np.asarray(self.parameters["output_range"])

    def _execute_test(self, test):
        test = self.descale(test.reshape(1, -1), self.irange).reshape(-1)

        # set matlab class directory, .m file
        matlab_engine.addpath(os.path.dirname(self.parameters["model_file"]))

        # make the matlab function from the input strings
        matlab_function_str = "matlab_engine.{}".format(os.path.basename(self.parameters["model_file"]))

        # create a callable function from the given string argument, from "model_file"
        # calls the matlab function with inputs list and get the output list
        # parse input parameters in any given dimension
        run_matlab_function = eval(matlab_function_str)([float(x) for x in test], nargout=1)

        return np.asarray([utpt for utpt in run_matlab_function])

