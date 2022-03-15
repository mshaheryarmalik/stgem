#!/usr/bin/python3
# -*- coding: utf-8 -*-

from stgem.algorithm import Model

class Random_Model(Model):
    """
    Implements a random test model which directly uses the sampling provided by
    the SUT.
    """

    def __init__(self, sut, parameters, logger=None):
        # TODO: describe the arguments
        super().__init__(sut, parameters, logger)

    def generate_test(self):
        """
        Generates a test for the SUT.
        """

        return self.sut.sample_input_space()

class Random_LHS_Model(Model):
    """
    Implements a random test model based on Latin hypercube design.
    """

    def __init__(self, sut, parameters, logger=None):
        super().__init__(sut, parameters, logger)

        try:
            from pyDOE import lhs
        except ImportError:
            raise

        if not "samples" in self.parameters:
            raise Exception("The 'samples' key must be provided for the algorithm for determining random sample size.")

        # Create the design immediately.
        self.random_tests = 2*(lhs(self.sut.idim, samples=self.samples) - 0.5)

        self.current = -1

    def generate_test(self):
        self.current += 1

        if self.current > len(self.random_tests):
            raise Exception("Random sample exhausted.")

        return self.random_tests[self.current]

