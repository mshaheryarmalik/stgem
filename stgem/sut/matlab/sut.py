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

class Matlab_Simulink_Signal(SUT):
    """
    Generic class for using Matlab Simulink models using signal inputs.
    """

    def __init__(self, parameters):
        super().__init__(parameters)

        # How often input signals are sampled for execution (in time units).
        self.steps = self.simulation_time // self.sampling_step

        if not os.path.exists(self.model_file + ".mdl") and not os.path.exists(self.model_file + ".slx"):
            raise Exception("Neither '{0}.mdl' nor '{0}.slx' exists.".format(self.model_file))

        self.MODEL_NAME = os.path.basename(self.model_file)
        # Initialize the Matlab engine (takes a lot of time).
        self.engine = matlab.engine.start_matlab()
        # The path for the model file.
        self.engine.addpath(os.path.dirname(self.model_file))
        # Get options for the model (takes a lot of time).
        model_opts = self.engine.simget(self.MODEL_NAME)
        # Set the output format of the model.
        # TODO: Should this be done for models other than AT?
        self.model_opts = self.engine.simset(model_opts, "SaveFormat", "Array")

    def _execute_test_simulink(self, timestamps, signals):
        """
        Execute a test with the given input signals.
        """

        # Setup the parameters for Matlab.
        simulation_time = matlab.double([0, timestamps[-1]])
        model_input = matlab.double(np.row_stack((timestamps, *signals)).T.tolist())

        # Run the simulation.
        out_timestamps, _, data = self.engine.sim(self.MODEL_NAME, simulation_time, self.model_opts, model_input, nargout=self.odim)

        timestamps_array = np.array(out_timestamps).flatten()
        data_array = np.array(data)

        # Reshape the data.
        result = np.zeros(shape=(self.odim, len(timestamps_array)))
        for i in range(self.odim):
            result[i] = data_array[:, i]

        return timestamps_array, result

    def _execute_test(self, timestamps, signals):
        return self._execute_test_simulink(timestamps, signals)

class Matlab_Simulink(Matlab_Simulink_Signal):
    """
    Generic class for using Matlab Simulink models using piecewise constant
    inputs. We assume that the input is a vector of numbers in [-1, 1] and that
    the first K numbers specify the pieces of the first signal, the next K
    numbers the second signal, etc. The number K is specified by the simulation
    time and the length of the time interval during which the signal must stay
    constant.
    """

    def __init__(self, parameters):
        try:
            super().__init__(parameters)
        except:
            raise

        # How many inputs we have for each signal.
        self.pieces = self.simulation_time // self.time_slice

    def initialize(self):
        # Redefine input dimension.
        self.signals = self.idim
        self.idim = self.idim*self.pieces

        # Redo input ranges for vector inputs.
        new = []
        for i in range(len(self.input_range)):
            for _ in range(self.pieces):
                new.append(self.input_range[i])
        self.input_range = new

    def _execute_test(self, test):
        """
        Execute the given test on the SUT.

        Args:
          test (np.ndarray): Array of floats with shape (1,N) or (N) with
                             N = self.idim.

        Returns:
          timestamps (np.ndarray): Array of shape (M, 1).
          signals (np.ndarray): Array of shape (self.odim, M).
        """

        test = self.descale(test.reshape(1, -1), self.input_range).reshape(-1)

        # Convert the test input to signals.
        idx = lambda t: int(t // self.time_slice) if t < self.simulation_time else self.pieces - 1
        signal_f = []
        for i in range(self.signals):
            signal_f.append(lambda t: test[i*self.pieces + idx(t)])
        timestamps = np.linspace(0, self.simulation_time, int(self.steps))
        signals = []
        for i in range(self.signals):
            signals.append(np.asarray([signal_f[i](t) for t in timestamps]))

        # Execute the test.
        return self._execute_test_simulink(timestamps, signals)

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

class Matlab(SUT):
    """
    Generic class for using Matlab m files with vector inputs.
    """

    """
    Currently we assume the following. The model_file parameter defines a
    Matlab function with the same name (function statement on the first line).
    It takes as its input floats specified by self.idim and returns self.odim
    values. If the function to be called does not fit this setting, this class
    needs to be extended.
    """

    def __init__(self, parameters):
        SUT.__init__(self, parameters)

        if not os.path.exists(self.model_file + ".m"):
            raise Exception("The file '{}.m' does not exist.".format(self.model_file))

        self.MODEL_NAME = os.path.basename(self.model_file)
        # Initialize the Matlab engine (takes a lot of time).
        self.engine = matlab.engine.start_matlab()
        # The path for the model file.
        self.engine.addpath(os.path.dirname(self.model_file))
        # Save the function into an object.
        self.matlab_func = getattr(self.engine, self.MODEL_NAME)

    def _execute_test(self, test):
        test = self.descale(test.reshape(1, -1), self.input_range).reshape(-1)

        # TODO: Add error handling in case of wrong input situations or Matlab
        #       errors.
        # Matlab does not like numpy data types, so we convert to floats.
        matlab_result = self.matlab_func(*(float(x) for x in test), nargout=self.odim)

        return np.asarray(matlab_result)

