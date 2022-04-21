#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, math

import numpy as np

try:
    import matlab
    import matlab.engine
except ImportError:
    raise Exception("Error importing Python Matlab engine.")

from stgem.sut import SUT, SUTResult

class Matlab_Simulink_Signal(SUT):
    """Generic class for using Matlab Simulink models using signal inputs."""

    def __init__(self, parameters):
        super().__init__(parameters)

        # How often input signals are sampled for execution (in time units).
        self.steps = self.simulation_time // self.sampling_step

        if not os.path.exists(self.model_file + ".mdl") and not os.path.exists(self.model_file + ".slx"):
            raise Exception("Neither '{0}.mdl' nor '{0}.slx' exists.".format(self.model_file))

    def setup_matlab(self):
        # As setting Matlab takes some time, we only spend this time if really
        # needed.
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
        # Figure out if we have a fixed-step solver or a variable-step solver.
        self.variable_step = self.model_opts["Solver"].lower().startswith("variablestep")

    def __del__(self):
        if hasattr(self, "engine"):
            self.engine.quit()

    def _execute_test_simulink(self, timestamps, signals):
        """Execute a test with the given input signals."""

        if not hasattr(self, "engine"):
            self.setup_matlab()

        # Setup the parameters for Matlab.
        simulation_time = matlab.double([0, timestamps[-1]])
        model_input = matlab.double(np.row_stack((timestamps, *signals)).T.tolist())

        """
        Output formats depends on the solver.

        Fixed-step solver:
        -----------------------------------------------------------------------
        Since the simulation steps are known in advance, Matlab can put the
        outputs in a matrix with columns time, output1, output2, ... This means
        that three values are returned timestamps, internal state, and the
        output matrix.

        Variable-step solver:
        -----------------------------------------------------------------------
        Since the simulation timesteps are not known in advance, the size of
        the matrix described is not known. Here Matlab returns 2 + outputs
        arrays. The first has the timesteps, the second is the internal state,
        and the remaining are the outputs.
        """

        # Run the simulation.
        if self.variable_step:
            output = self.engine.sim(self.MODEL_NAME, simulation_time, self.model_opts, model_input, nargout=self.odim + 2)
            out_timestamps = output[0]

            result = np.zeros(shape=(self.odim, len(output[0])))
            for i in range(self.odim):
                result[i] = np.asarray(output[2+i]).flatten()
        else:
            out_timestamps, _, data = self.engine.sim(self.MODEL_NAME, simulation_time, self.model_opts, model_input, nargout=self.odim)
            data = np.asarray(data)

            # Reshape the data.
            result = np.zeros(shape=(self.odim, len(out_timestamps)))
            for i in range(self.odim):
                result[i] = data[:, i]

        output_timestamps = np.array(out_timestamps).flatten()

        return SUTResult(signals, result, timestamps, output_timestamps, None)

    def _execute_test(self, timestamps, signals):
        return self._execute_test_simulink(timestamps, signals)

class Matlab_Simulink(Matlab_Simulink_Signal):
    """Generic class for using Matlab Simulink models using piecewise constant
    inputs. We assume that the input is a vector of numbers in [-1, 1] and that
    the first K1 numbers specify the pieces of the first signal, the next K2
    numbers the second signal, etc. The numbers K1, K2, ... are determined by
    the simulation time and the lengths of time intervals during which the
    signal must stay constant. This is controlled by the SUT parameter
    time_slices which is a list of floats."""

    def __init__(self, parameters):
        try:
            super().__init__(parameters)
        except:
            raise

        if not "time_slices" in self.parameters:
            raise Exception("Parameter 'time_slices' must be defined for piecewise constant signal inputs.")
        if not "simulation_time" in self.parameters:
            raise Exception("Parameter 'simulation_time' must be defined for piecewise constant signal inputs.")
        if not "sampling_step" in self.parameters:
            raise Exception("Parameter 'sampling_step' must be defined for piecewise constant signal inputs.")

        # How often input signals are sampled for execution (in time units).
        self.steps = int(self.simulation_time / self.sampling_step)
        # How many inputs we have for each input signal.
        self.pieces = [math.ceil(self.simulation_time / time_slice) for time_slice in self.time_slices]

    def setup(self, budget):
        super().setup(budget)

        if not len(self.time_slices) == self.idim:
            raise Exception("Expected {} time slices, found {}.".format(self.idim, len(self.time_slices)))

        self.N_signals = self.idim
        self.idim = sum(self.pieces)

        self.descaling_intervals = []
        for i in range(len(self.input_range)):
            for _ in range(self.pieces[i]):
                self.descaling_intervals.append(self.input_range[i])

    def _execute_test(self, test):
        test = self.descale(test.reshape(1, -1), self.descaling_intervals).reshape(-1)

        # Common timestamps to all input signals.
        timestamps = np.array([i*self.sampling_step for i in range(self.steps + 1)])
        # Signals.
        signals = np.zeros(shape=(self.N_signals, len(timestamps)))
        offset = 0
        for i in range(self.N_signals):
            idx = lambda t: int(t // self.time_slices[i]) if t < self.simulation_time else self.pieces[i] - 1
            signal_f = lambda t: test[offset + idx(t)]
            signals[i] = np.asarray([signal_f(t) for t in timestamps])
            offset += self.pieces[i]

        # Execute the test.
        return self._execute_test_simulink(timestamps, signals)

    def execute_random_test(self):
        test = self.sample_input_space()
        tr = self.execute_test(test)
        return tr

class Matlab(SUT):
    """Generic class for using Matlab m files."""

    """
    Currently we assume the following. The model_file parameter defines a
    Matlab function with the same name (function statement on the first line).
    It takes as its input a sequence of floats or signals. This is specified by
    setting the parameter 'input_type' to 'vector' or 'signal'. Similarly the
    output of the Matlab function is specified by setting the parameter
    'output_type' to have value 'vector' or 'signal'.

    Currently we assume that if the Matlab function expects signals as inputs,
    then the Matlab function's argument is U, a data matrix such that its first
    column corresponds to the timestamps, second column to the first signal
    etc.

    If an initializer Matlab program needs to be run before calling the actual
    function, this is accomplished by giving the program file (without the
    extension .m) as the parameter 'init_model_file'. This program is called
    with nargout=0 and we assume that it needs to be run only once.
    """

    def __init__(self, parameters):
        super().__init__(parameters)

        if not os.path.exists(self.model_file + ".m"):
            raise Exception("The file '{}.m' does not exist.".format(self.model_file))

        if "init_model_file" in self.parameters and not os.path.exists(self.init_model_file + ".m"):
            raise Exception("The file '{}.m' does not exist.".format(self.init_model_file))

        if not "input_type" in self.parameters:
            raise Exception("Matlab call input type not specified.")
        if not self.input_type.lower() in ["vector", "piecewise constant signal", "signal"]:
            raise Exception("Unknown Matlab call input type '{}'.".format(self.input_type))

        if not "output_type" in self.parameters:
            raise Exception("Matlab call output type not specified.")
        if not self.output_type.lower() in ["vector", "signal"]:
            raise Exception("Unknown Matlab call output type '{}'.".format(self.output_type))

        if self.input_type == "piecewise constant signal":
            if not "time_slices" in self.parameters:
                raise Exception("Parameter 'time_slices' must be defined for piecewise constant signal inputs.")
            if not "simulation_time" in self.parameters:
                raise Exception("Parameter 'simulation_time' must be defined for piecewise constant signal inputs.")
            if not "sampling_step" in self.parameters:
                raise Exception("Parameter 'sampling_step' must be defined for piecewise constant signal inputs.")

            # How often input signals are sampled for execution (in time units).
            self.steps = int(self.simulation_time / self.sampling_step)
            # How many inputs we have for each input signal.
            self.pieces = [math.ceil(self.simulation_time / time_slice) for time_slice in self.time_slices]

    def setup_matlab(self):
        # As setting Matlab takes some time, we only spend this time if really
        # needed.
        self.MODEL_NAME = os.path.basename(self.model_file)
        self.INIT_MODEL_NAME = os.path.basename(self.init_model_file) if "init_model_file" in self.parameters else None

        # Initialize the Matlab engine (takes a lot of time).
        self.engine = matlab.engine.start_matlab()
        # The paths for the model files.
        self.engine.addpath(os.path.dirname(self.model_file))
        if self.INIT_MODEL_NAME is not None:
            self.engine.addpath(os.path.dirname(self.init_model_file))
        # Save the function into an object.
        self.matlab_func = getattr(self.engine, self.MODEL_NAME)

        # Run the initializer program.
        if self.INIT_MODEL_NAME is not None:
            init = getattr(self.engine, self.INIT_MODEL_NAME)
            init(nargout=0)

    def setup(self, budget):
        super().setup(budget)

        # Adjust the SUT parameters if the input is a piecewise constant
        # signal.
        if self.input_type == "piecewise constant signal":
            if not len(self.time_slices) == self.idim:
                raise Exception("Expected {} time slices, found {}.".format(self.idim, len(self.time_slices)))

            self.N_signals = self.idim
            self.idim = sum(self.pieces)

            self.descaling_intervals = []
            for i in range(len(self.input_range)):
                for _ in range(self.pieces[i]):
                    self.descaling_intervals.append(self.input_range[i])

    def __del__(self):
        if hasattr(self, "engine"):
            self.engine.quit()

    def _execute_test(self, *args, **kwargs):
        # TODO: Add error handling in case of wrong input or Matlab errors.
        if not hasattr(self, "engine"):
            self.setup_matlab()

        if self.input_type == "vector":
            test = args[0]
            test = self.descale(test.reshape(1, -1), self.input_range).reshape(-1)

            if self.output_type == "vector":
                # Matlab does not like numpy data types, so we convert to floats.
                matlab_result = self.matlab_func(*(float(x) for x in test), nargout=self.odim)
                matlab_result = np.asarray(matlab_result)

                return SUTResult(test, matlab_result, None, None, None)
            else:
                matlab_result = self.matlab_func(*(float(x) for x in test), nargout=2)
                output_timestamps = np.asarray(matlab_result[0]).flatten()
                data = np.asarray(matlab_result[1])

                # Reshape the data.
                output_signals = np.zeros(shape=(self.odim, len(output_timestamps)))
                for i in range(self.odim):
                    output_signals[i] = data[:, i]

                return SUTResult(test, output_signals, None, output_timestamps, None)
        else:
            # If we have a piecewise constant signal, convert the input vector
            # to a constant signal.
            if self.input_type == "piecewise constant signal":
                test = args[0]
                test = self.descale(test.reshape(1, -1), self.descaling_intervals).reshape(-1)

                # Common timestamps to all input signals.
                timestamps = np.array([i*self.sampling_step for i in range(self.steps + 1)])
                # Signals.
                signals = np.zeros(shape=(self.N_signals, len(timestamps)))
                offset = 0
                for i in range(self.N_signals):
                    idx = lambda t: int(t // self.time_slices[i]) if t < self.simulation_time else self.pieces[i] - 1
                    signal_f = lambda t: test[offset + idx(t)]
                    signals[i] = np.asarray([signal_f(t) for t in timestamps])
                    offset += self.pieces[i]
            else:
                timestamps = args[0]
                signals = args[1]
            
            # Prepare the input signals to a format expected by Matlab; see
            # above.
            model_input = matlab.double(np.column_stack((timestamps, *signals)).tolist())

            if self.output_type == "vector":
                matlab_result = self.matlab_func(model_input, nargout=self.odim)

                return SUTResult(signals, matlab_result, timestamps, None, None)
            else:
                matlab_result = self.matlab_func(model_input, nargout=2)
                output_timestamps = np.asarray(matlab_result[0]).flatten()
                data = np.asarray(matlab_result[1])

                # Reshape the data.
                output_signals = np.zeros(shape=(self.odim, len(output_timestamps)))
                for i in range(self.odim):
                    output_signals[i] = data[:, i]

                return SUTResult(signals, output_signals, timestamps, output_timestamps, None)

