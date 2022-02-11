#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from sut.sut import SUT

class PythonFunction(SUT):

    def __init__(self,parameters):
        SUT.__init__(self,parameters)

        self.irange = np.asarray(self.parameters["input_range"])
        self.orange = np.asarray(self.parameters["output_range"])
        self.function= self.parameters["function"]
        self.idim = len( self.irange )
        self.odim = len( self.orange )

    def _execute_test(self, test):
        test = self.descale(test.reshape(1, -1), self.irange).reshape(-1)
        output=self.function(test)

        return np.asarray(output)




