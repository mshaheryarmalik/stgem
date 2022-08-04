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

