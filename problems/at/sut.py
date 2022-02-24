#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import numpy as np

try:
    import matlab
    import matlab.engine
except ImportError:
    raise Exception("Error importing Python Matlab engine for AT.")

from stgem.sut import SUT

class AT(SUT):
    """
    Class for the automatic transmission (AT) SUT. Currently the time domain for
    the input signals is 30 units and it is split into 6 constant pieces.
    """

    def __init__(self, parameters):
        super().__init__(parameters)

        self.parameters = parameters

        # TODO: Right now the values below are fixed as in the Fainekos et al.
        #       paper. We should consider making them configurable.

        # The total time domain for the signal.
        ##self.time = 30
        self.time = self.parameters["algorithm_parameters"]["time"]
        # How many time units a signal must stay constant. This determines how
        # many pieces we have for each input signal.
        ##self.minimum_time = 5
        self.minimum_time = self.parameters["algorithm_parameters"]["minimum_time"]
        self.pieces = self.time // self.minimum_time
        # How often input signals are sampled for execution (in time units).
        ##self.sampling_step = 0.2
        self.sampling_step = self.parameters["algorithm_parameters"]["sampling_step"]
        self.steps = self.time // self.sampling_step

        self.idim = 2*self.pieces
        ##self.odim = 3
        self.odim = self.odim = self.parameters["algorithm_parameters"]["odim"]

        # The ranges for the input signals come from ARCH COMP 2020.
        throttle_range = (0, 100)
        brake_range = (0, 325)
        self.irange = np.asarray([throttle_range for _ in range(self.pieces)] + [brake_range for _ in range(self.pieces)])

        # TODO: Figure out correct output ranges.
        speed_range = (0, 200)
        rpm_range = (0, 7000)
        gear_range = (0, 4)
        self.orange = np.asarray([speed_range, rpm_range, gear_range])

        if not os.path.exists(self.model_file + ".mdl"):
            raise Exception("Model file '{}.mdl' does not exist.".format(self.model_file))
        self.MODEL_NAME = os.path.basename(self.model_file)
        # Initialize the Matlab engine (takes a lot of time).
        self.engine = matlab.engine.start_matlab()
        # The path for the model file.
        self.engine.addpath(os.path.dirname(self.model_file))
        # Get options for the model (takes a lot of time).
        model_opts = self.engine.simget(self.MODEL_NAME)
        # Set the output format of the model.
        self.model_opts = self.engine.simset(model_opts, "SaveFormat", "Array")

    def _execute_test_at(self, timestamps, throttle, brake):
        """
        Execute a test with the given input signals.
        """

        # Setup the parameters for Matlab.
        simulation_time = matlab.double([0, timestamps[-1]])
        model_input = matlab.double(np.row_stack((timestamps, throttle, brake)).T.tolist())

        # Run the simulation.
        out_timestamps, _, data = self.engine.sim(self.MODEL_NAME, simulation_time, self.model_opts, model_input, nargout=3)

        timestamps_array = np.array(out_timestamps).flatten()
        data_array = np.array(data)

        # Reshape the data.
        result = np.zeros(shape=(3, len(timestamps_array)))
        for i in range(3):
            result[i] = data_array[:, i]

        return timestamps_array, result

    def _execute_test(self, test):
        """
        Execute the given test on the SUT.

        Args:
          test (np.ndarray): Array with shape (1,N) or (N) with N = 2*self.pieces
                             floats.

        Returns:
          timestamps (np.ndarray): Array of shape (M, 1).
          signals (np.ndarray): Array of shape (3, M).
        """

        test = self.descale(test.reshape(1, -1), self.irange).reshape(-1)

        # Convert the test vector to two functions which take time as input and
        # return the signal values.
        idx = (lambda t: int(t // self.minimum_time) if t < self.time else self.pieces - 1)
        signals = [lambda t: test[idx(t)], lambda t: test[self.pieces + idx(t)]]

        # Setup the signals.
        timestamps = np.linspace(0, self.time, int(self.steps))
        throttle = [signals[0](t) for t in timestamps]
        brake = [signals[1](t) for t in timestamps]

        # Execute the test.
        return self._execute_test_at(timestamps, throttle, brake)

    def execute_random_test(self):
        """
        Execute a random tests and return it and its output.

        Returns:
          test (np.ndarray): Array of shape (2*self.pieces) of floats in [-1, 1].
          timestamps (np.ndarray): Array of shape (N, 1).
          signals (np.ndarray): Array of shape (3, N).
        """

        test = self.sample_input_space()
        timestamps, signals = self.execute_test(test)
        return test, timestamps, signals

    def sample_input_space(self):
        """
        Return a sample (test) from the input space.

        Returns:
          test (np.ndarray): Array of shape (2*self.pieces) of floats in [-1, 1].
        """

        return np.random.uniform(-1, 1, size=2 * self.pieces)

