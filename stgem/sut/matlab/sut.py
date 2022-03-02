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
    Implements a certain mathematical function as a SUT.

    The function is from
    L. Mathesen, G. Pedrielli, and G. Fainekos. Efficient optimization-based
    falsification of cyber-physical systems with multiple conjunctive
    requirements. In 2021 IEEE 17th International Conference on Automation
    Science and Engineering (CASE), pages 732–737, 2021.

    We fix the input domain of the function to be [-15, 15]^3. The tests are
    3-tuples of numbers in [-1, 1] which are scaled to [-15, 15] internally.
    """

    def __init__(self, parameters):
        SUT.__init__(self, parameters)

        self.parameters = parameters

        self.idim = len(self.parameters["input_range"])
        self.odim = len(self.parameters["output_range"])
        self.irange = np.asarray(self.parameters["input_range"])
        self.orange = np.asarray(self.parameters["output_range"])

    def _execute_test(self, test):
        # print("unscaled",test)
        test = self.descale(test.reshape(1, -1), self.irange).reshape(-1)
        # print("descaled", test)

        # set matlab class directory, .m file
        matlab_engine.addpath(os.path.dirname(self.parameters["model_file"]))

        # make the matlab function from the input strings
        matlab_function_str = "matlab_engine.{}".format(os.path.basename(self.parameters["model_file"]))

        # create a callable function from the given string argument, from "model_file"
        # calls the matlab function with inputs list and get the output list
        # parse input parameters in any given dimension
        run_matlab_function = eval(matlab_function_str)([float(x) for x in test], nargout=1)

        # h1 = 305-100*(math.sin(x1/3)+math.sin(x2/3)+math.sin(x3/3))
        # h2 = 230-75*(math.cos(x1/2.5+15)+math.cos(x2/2.5+15)+math.cos(x3/2.5+15))
        # h3 = (x1-7)**2+(x2-7)**2+(x3-7)**2 - (math.cos((x1-7)/2.75) + math.cos((x2-7)/2.75) + math.cos((x3-7)/2.75))

        return np.asarray([utpt for utpt in run_matlab_function])

    def _min_distance(self, tests, x):
        # We use the Euclidean distance.
        tests = np.asarray(tests)
        d = np.linalg.norm(tests - x, axis=1)
        return min(d)

    def execute_random_test(self):
        test = self.sample_input_space()
        return test, self._execute_test(test)

    def sample_input_space(self):
        return np.random.uniform(-1, 1, size=(1, self.idim))
