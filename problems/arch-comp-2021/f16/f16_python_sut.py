#!/usr/bin/python3
# -*- coding: utf-8 -*-

import subprocess

import numpy as np

from stgem.sut import SUT, SUTResult

class F16GCAS(SUT):
    """SUT for the Python version of the F16 problem."""

    def __init__(self, parameters):
        SUT.__init__(self, parameters)

        self.initial_altitude = 4040

    def _execute_test(self, test):
        test = self.descale(test.reshape(1, -1), self.input_range).reshape(-1)

        output = subprocess.run(["problems/arch-comp-2021/f16/AeroBenchVVPython/check_gcas_v1.sh", str(self.initial_altitude), str(test[0]) , str(test[1]), str(test[2]) ], capture_output=True)

        # Altitude on the last line.
        try:
            v = float(str(output.stdout).split("\\n")[-2])
        except:
            print(output.stdout)
            # FIXME?
            v = 4040

        return SUTResult(test, np.asarray([v]), None, None, None)

