#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from stgem.sut import SUT, SUTOutput


class ModelBasedSUT(SUT):
    """
    A SUT which uses one or more models to produce its output

    We expect that the models are already trained, for example by loading weights from model.load_from_file
    This can be useful if we want to explore what a model has learned after testing a system
    """

    def __init__(self, models, parameters=None):
        super().__init__(parameters)

        # we need at least one model
        assert len(models) >= 1
        self.models = models

        # we get the input dimension from the first model
        self.idim = self.models[0].get_input_dimension()
        # the output dimension is the number of provided models
        self.odim = len(self.models)
        # All models have the same input and output ranges
        self.input_range  = [[-1, 1]] * self.idim
        self.output_range = [[0, 1]] * self.odim

    def _execute_test(self, test):
        # This is not necessary
        #denormalized = self.descale(test.inputs.reshape(1, -1), self.input_range).reshape(-1)
        denormalized=test.inputs
        output = []
        error = None

        # use the models to generate the outputs
        for model in self.models:
            output.append(model.predict_objective(np.array([denormalized]))[0])

        test.input_denormalized = denormalized

        return SUTOutput(np.asarray(output), None, error)
