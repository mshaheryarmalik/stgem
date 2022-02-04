#!/usr/bin/python3
# -*- coding: utf-8 -*-

from algorithm.model import Model

class Random_Model(Model):
    """
    Implements the random test model.
    """

    def __init__(self, sut, parameters, logger=None):
        # TODO: describe the arguments
        super().__init__(sut, parameters, logger)

    def generate_test(self):
        """
        Generates a test for the SUT.
        """

        return self.sut.sample_input_space()
