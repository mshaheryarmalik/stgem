#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
# SUT
A system under test (SUT) is represented by a class that inherits from the base class `SUT`. A SUT is a deterministic function that maps an input to an output. Both inputs and outputs can be either vectors or discrete signals. The execution of this function is achieved by the method `execute_test` which takes a `SUTInput` object as an input and returns a `SUTOutput` object.

## SUT Initialization
The `__init__` method of a SUT by convention takes as an argument a dictionary `parameters` which is supposed to contain all static parameters (strings, integers, etc.) the SUT needs. Other arguments can be specified, but we recommend minimalism here. The base class saves the input dictionary `parameters` as `x.parameters`. For convenience, we allow writing `x.foo` in place of `x.parameters["foo"]`, and it is important for this functionality that an inheriting class calls the parent class `__init__` method. Since some SUTs can be resource-heavy, we recommend to design SUT objects that are reusable. This means that they should not depend on external objective functions, algorithms, etc.

Each SUT has a `setup` method (taking no arguments) which is called in an STGEM object just before the generator is run. The purpose is to provide a two-step initialization where the user can alter the SUT object after it has been initialized but before it is being setup for use. We recommend that all resource-heavy initialization goes into the `setup` method. In this way several SUT objects can be initialized in advance without significant resource penalties. The `setup` method needs to be idempotent meaning that calling it several times results in the same outcome. It is mandatory to call the parent class `setup` method.

## Common SUT Parameters and Attributes
There are few values in the `parameters` dictionary which are common to all SUTs. These are related to inputs and outputs. They are as follows:

* `inputs`: `int` or list of `str`
* `input_range`: list of `[float, float]` or `None`
* `outputs`: `int` or list of `str`
* `output_range`: list of `[float, float]`or `None`

Let us describe the input parameters only; outputs are similar. The parameter `inputs` specifies the inputs for the SUT. The value can be an integer or a list of strings. If a list of strings is given, say `["speed", "brake"]`, we have two inputs with names `"speed"` and `"brake"`. If a number is used, say `2`, then there are two inputs with default names `"i0"` and `"i1"` etc. The number of inputs is the number of signals for SUTs with signal inputs and the number of components for SUTs with vector inputs.

The parameter `input_range` specifies the ranges for the inputs as a list of 2-tuples of floats, for example `[[0, 100], [0, 325]]`. It is allowed to use `None` as an interval to indicate an unknown value. If the parameter `inputs` is omitted, it is inferred from this parameter. If `inputs` is specified, then intervals can be omitted and the value `None` is automatically used for missing values.

Neither `inputs` nor `input_range` needs to be specified in the `parameters` dictionary if they are set in the SUT class `__init__` method.

In addition to the above parameters, each SUT has attributes `idim` and `odim` available after `setup` has been called. These simply contain number of inputs (dimension of a vector or number of signals) and number of outputs (dimension of a vector or number of signals) respectively. Input and output types of a SUT can be signaled using the strings `"vector"` and `"signal"` as values for the SUT attributes `input_type` and `output_type`. By default `input_type` and `output_type` are `None`, and they must be explicitly set in SUT initialization if desired.

## Common SUT Methods

### Normalization and Denormalization
The SUT methods `scale`, `scale_signal`, and `descale` provide a way to map points and signals between intervals.

### Variable Ranges
The method `variable_range` returns the range of a variable (input or output) given its name, that is, `variable_range("speed")` returns the range of the variable with name `"speed"`.

## Input Normalization
We take the convention that the machine learning related parts of our code deal exclusively with normalized inputs. This means that the numerical values in the `SUTInput` object `inputs` attribute (see below) must be given as elements of the interval [-1, 1] meaning that input vectors are lists of numbers in [-1, 1] and input signals are functions taking values in [-1, 1]. Obviously the SUT itself may deal with other ranges and needs to denormalize. This is handled by the SUT internally based on the `input_range` specified when initializing and setuping the SUT.

TODO: Decide what happens when a range is unknow (`None`). Currently the behavior is undefined.

## Inputs and Outputs
The number of inputs and outputs (dimension of a vector or number of signals) and optionally input and output ranges need to be specified for every SUT; see above. The inputs and outputs are handled via the objects `SUTInput` and `SUTOutput`.

The `SUTInput` object has three attributes: `inputs`, `input_denormalized`, and `input_timestamps`. If the input is of vector type, then the vector is defined as a 1D numpy array in `inputs` and `input_timestamps` is `None`. If the input is of signal type, then `inputs` is a 2D numpy array whose each row determines signal values and the corresponding timestamps (common to all signals) are given as a 1D numpy array in `input_timestamps`. The attribute `input_denormalized` is to contain the denormalized version of `inputs`. As the denormalization is internal to the SUT, the convention is that `input_denormalized` is `None` when given to the method `execute_test` of a SUT and is available after `execute_test` has finished. The denormalized input is mainly available for debugging purposes, so it can be set back to `None` in order to conserve memory.

The `SUTOutput` object has three attributes: `outputs`, `output_timestamps` and `error`. The `error` attribute is a string describing what error occurred during the SUT execution (if any); if there was no error, its value is `None`. The attributes `outputs` and `output_timestamps` behave as `inputs` and `input_timestamps` above. Notice that the output numerical values are unnormalized, it is up to the user to decide whether to normalize based on SUT output ranges or something else.

## Input Validity
A SUT may have a notion of a valid test, that is, not all elements of its input space are considered executable. For validation, the SUT should implement the method `validity` which takes a SUTInput object as an argument. It should return 0 for invalid tests and 1 for valid tests. The default implementation always returns 1.

## Exceptions
TODO
"""

"""
NOTICE: We support different ranges for each output value, but doing so is not
always a good idea. This is because most algorithms we use directly compare the
objective function values in [0, 1], so they should in some sense be
comparable.
"""

from dataclasses import dataclass

import numpy as np

from stgem.performance import PerformanceData

@dataclass
class SUTInput:
    inputs: ...
    input_denormalized: ...
    input_timestamps: ...

@dataclass
class SUTOutput:
    outputs: ...
    output_timestamps: ...
    error: ...

class SearchSpace:
    """
    Each SUT has a natural input space defined by its number of inputs, input types, and input ranges.
    The purpose of the SearchSpace object is to describe this input space and provide methods for it (random sampling etc.).
    However, random sampling, for example, modifies an internal state, so not all functionalities can be provided by the SUT itself as we want SUT objects to be reusable.
    The point of a SearchSpace object is thus separate certain functionalities from SUTs in order to preserve reusability.
    """
    def __init__(self):
        self.sut = None
        self.input_dimension = 0
        self.output_dimension = 0
        self.rng = None

    def setup(self, sut, objectives, rng):
        self.sut = sut
        self.input_dimension = self.sut.idim
        self.output_dimension = self.sut.odim
        self.odim = len(objectives)
        self.rng = rng

    def is_valid(self, test) -> bool:
        # This is here until valid tests are changed to preconditions. This
        # line ensures that model-based SUTs work and can be pickled.
        if self.sut is None: return True
        return self.sut.validity(test)

    def sample_input_space(self):
        return self.rng.uniform(-1, 1, size=self.input_dimension)

class SUT:
    """Base class implementing a system under test. """

    default_parameters = {}

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = {}

        # merge deafult_parameters and parameters, the later takes priority if a key appears in both dictionaries
        # the result is a new dictionary
        self.parameters = self.default_parameters | parameters

        if not "input_type" in self.parameters:
            self.parameters["input_type"] = None
        if not "output_type" in self.parameters:
            self.parameters["output_type"] = None

        self.perf = PerformanceData()
        self.base_has_been_setup = False

    def __getattr__(self, name):
        if "parameters" in self.__dict__:
            if name in self.parameters:
                return self.parameters.get(name)

        raise AttributeError(name)

    def setup(self):
        """Setup the budget and perform steps necessary for two-step
        initialization. Derived classes should always call this super class
        setup method."""

        # We skip setup if it has been done before since inheriting classes
        # may alter idim, odim, ranges, etc.
        if self.base_has_been_setup: return

        # Infer dimensions and names for inputs and outputs from impartial
        # information.

        # If self.inputs exists and is an integer, transform it into default
        # input names i1, ...iN where N is this integer. This also determines
        # idim if unset.
        if hasattr(self, "inputs") and isinstance(self.inputs, int):
            if not hasattr(self, "idim"):
                self.idim = self.inputs
            self.inputs = ["i{}".format(i) for i in range(self.inputs)]

        # If idim is not set, it can be inferred from input names (a list of
        # names) or input ranges.
        if hasattr(self, "idim"):
            # idim set already, set default input names if necessary.
            if not hasattr(self, "inputs"):
                self.inputs = ["i{}".format(i) for i in range(self.idim)]
        else:
            # idim can be inferred from input names, if defined.
            if hasattr(self, "inputs"):
                self.idim = len(self.inputs)
            else:
                # idim can be inferred from input ranges. Otherwise we do not
                # know what to do.
                if not hasattr(self, "input_range"):
                    raise Exception("SUT input dimension not defined and cannot be inferred.")
                self.idim = len(self.input_range)
                self.inputs = ["i{}".format(i) for i in range(self.idim)]

        # The same as above for outputs.
        if hasattr(self, "outputs") and isinstance(self.outputs, int):
            if not hasattr(self, "odim"):
                self.odim = self.outputs
            self.outputs = ["o{}".format(i) for i in range(self.outputs)]

        if hasattr(self, "odim"):
            if not hasattr(self, "outputs"):
                self.outputs = ["o{}".format(i) for i in range(self.odim)]
        else:
            if hasattr(self, "outputs"):
                self.odim = len(self.outputs)
            else:
                if not hasattr(self, "output_range"):
                    raise Exception("SUT output dimension not defined and cannot be inferred.")
                self.odim = len(self.output_range)
                self.outputs = ["o{}".format(i) for i in range(self.odim)]

        # Setup input and output ranges and fill unspecified input and output
        # ranges with Nones.
        if not hasattr(self, "input_range"):
            self.input_range = []
        if not isinstance(self.input_range, list):
            raise Exception("The input_range attribute of the SUT must be a Python list.")
        self.input_range += [None for _ in range(self.idim - len(self.input_range))]
        if not hasattr(self, "output_range"):
            self.output_range = []
        if not isinstance(self.output_range, list):
            raise Exception("The output attribute of the SUT must be a Python list.")
        self.output_range += [None for _ in range(self.odim - len(self.output_range))]

        self.base_has_been_setup = True

    def variable_range(self, var_name):
        """Return the range for the given variable (input or output)."""

        # NOTICE: Attributes might not exist unless the setup method has been called.
        if hasattr(self, "output_range"):
            for n, v in enumerate(self.outputs):
                if var_name == v:
                    return self.output_range[n]
        if hasattr(self, "input_range"):
            for  n, v in enumerate(self.inputs):
                if var_name == v:
                    return self.input_range[n]

        raise Exception("No variable '{}'.".format(var_name))

    def scale(self, x, intervals, target_A=-1, target_B=1):
        """
        Return a scaled x where the components of x with the specified
        intervals are scaled to the interval [A, B] (default [-1, 1]). If an
        interval is None, then no scaling is done.
        """

        if len(intervals) < x.shape[1]:
            raise Exception("Not enough intervals ({}) for scaling a vector of length {}.".format(len(intervals), x.shape[1]))

        y = np.zeros_like(x)
        for i in range(x.shape[1]):
            if intervals[i] is not None:
                A = intervals[i][0]
                B = intervals[i][1]
                C = (target_B-target_A)/(B-A)
                D = target_A - C*A
                y[:,i] = C*x[:,i] + D
            else:
                y[:,i] = x[:,i]

        return y

    def scale_signal(self, signal, interval, target_A=-1, target_B=1):
        """
        Scales the input signal whose values are in the given interval to the
        specified interval [A, B] (default [-1, 1]). If the interval is None,
        then no scaling is done.
        """

        y = np.asarray(signal)
        if interval is not None:
            A = interval[0]
            B = interval[1]
            C = (target_B-target_A)/(B-A)
            D = target_A - C*A
            return C*y + D
        else:
            return y

    def descale(self, x, intervals, A=-1, B=1):
        """
        Return a scaled x where the components of x in [A, B] (default [-1, 1])
        are scaled to the given intervals. If an interval is None, then no
        scaling is done.
        """

        if len(intervals) < x.shape[1]:
            raise Exception("Not enough intervals ({}) for descaling a vector of length {}.".format(len(intervals), x.shape[1]))

        y = np.zeros_like(x)
        for i in range(x.shape[1]):
            if intervals[i] is not None:
                target_A = intervals[i][0]
                target_B = intervals[i][1]
                C = (target_B-target_A)/(B-A)
                D = target_A - C*A
                y[:,i] = C*x[:,i] + D
            else:
                y[:,i] = x[:,i]

        return y

    def _execute_test(self, test: SUTInput) -> SUTOutput:
        raise NotImplementedError()

    def execute_test(self, test: SUTInput) -> SUTOutput:
        # Check for correct input type if specified.
        if self.input_type is not None:
            if self.input_type == "vector":
                if test.input_timestamps is not None or len(test.inputs.shape) > 1:
                    raise Exception("Signal input given for vector input SUT.")
            elif self.input_type == "signal":
                if test.input_timestamps is None or len(test.inputs.shape) == 1:
                    raise Exception("Vector input given for vector input SUT.")

        # TODO: Check for output.error.
        try:
            output = self._execute_test(test)
        except:
            raise

        # Check for correct output type if specified.
        if self.output_type is not None:
            if self.output_type == "vector":
                if output.output_timestamps is not None or len(output.outputs.shape) > 1:
                    raise Exception("Signal output for vector output SUT.")
            elif self.output_type == "signal":
                if output.output_timestamps is None or len(output.outputs.shape) == 1:
                    raise Exception("Vector output for signal output SUT.")
        
        return output

    def validity(self, test: SUTInput) -> int:
        """Basic validator which deems all tests valid."""

        return 1

