#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy as np

from stgem.sut import SUT

class PythonFunction(SUT):

    def __init__(self,parameters):
        SUT.__init__(self,parameters)

        if "input_range" in self.parameters:
            self.irange = np.asarray(self.parameters["input_range"])
        if "output_range" in self.parameters:
            self.orange = np.asarray(self.parameters["output_range"])

        self.function= self.parameters["function"]

        # Look up input and output ranges from function annotations
        for k,v in self.function.__annotations__.items():
            if k == "return":
                self.orange = np.asarray(v)
            else:
                self.irange = np.asarray(v)

        self.idim = len( self.irange )
        self.odim = len( self.orange )

    def _execute_test(self, test):
        test = self.descale(test.reshape(1, -1), self.irange).reshape(-1)
        output=self.function(test)

        return np.asarray(output)




