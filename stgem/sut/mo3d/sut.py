#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math

import numpy as np

from stgem.sut import SUT,  SUTResult

class MO3D(SUT):
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

    def __init__(self,parameters=None):
        SUT.__init__(self,parameters)

        self.input_range = [[-15, 15], [-15, 15], [-15, 15]]
        self.output_range = [[0, 350], [0, 350], [0, 350]]
        self.inputs=["i0","i1","i2"]
        self.outputs=["o0","o1","o2"]
        self.idim=3
        self.odim=3

    def execute_test(self, test):
        #print("unscaled",test)
        test = self.descale(test.reshape(1, -1), self.input_range).reshape(-1)
        #print("descaled", test)

        x1 = test[0]
        x2 = test[1]
        x3 = test[2]

        h1 = 305-100*(math.sin(x1/3)+math.sin(x2/3)+math.sin(x3/3))
        h2 = 230-75*(math.cos(x1/2.5+15)+math.cos(x2/2.5+15)+math.cos(x3/2.5+15))
        h3 = (x1-7)**2+(x2-7)**2+(x3-7)**2 - (math.cos((x1-7)/2.75) + math.cos((x2-7)/2.75) + math.cos((x3-7)/2.75))

        return SUTResult(test,np.asarray([h1, h2, h3]),None,None, None)

