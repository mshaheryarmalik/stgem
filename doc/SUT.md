# SUT
A system under test (SUT) is represented by a class that inherits from the base class SUT. A SUT is a deterministic function that maps an input to an output. Both inputs and outputs can be either vectors or discrete signals. The execution of this function is achieved by the method `execute_test` which takes a SUTInput as an input and returns a SUTResult object.

## SUT Initialization
The `__init__` method of a SUT by convention takes as an argument a dictionary `parameters` which is supposed to contain all static (strings, integers, etc.) parameters the SUT needs. Other arguments can be specified, but we recommend minimalism here. The base class saves the input dictionary `parameters` as `x.parameters`. For convenience, we allow writing `x.foo` in place of `x.parameters["foo"]`, and it is important for this functionality that an inheriting class calls the parent class `__init__` method. Since some SUTs can be resource-heavy, we recommend to design SUT objects that are reusable. This means that they should not depend on external objective functions, algorithms, etc.

Each SUT has a `setup` method (taking no arguments) which is called in an STGEM object just before the generator is run. The purpose is to provide a two-step initialization where the user can alter the SUT object after it has been initialized but before it is being setup for use. We recommend that all resource-heavy initialization goes into the `setup` method. In this way several SUT objects can be initialized in advance without significant resource penalties. The `setup` method needs to be idempotent meaning that calling it several times results in the same outcome. It is mandatory to call the parent class `setup` method.

## Common SUT Parameters and Attributes
There are few values in the `parameters` dictionary which are common to all SUTs. These are related to inputs and outputs. They are as follows:

* `inputs`: int or list of strings
* `input_range`: list of `[float, float]` or `None`
* `outputs`: int or list of strings
* `output_range`: list of `[float, float]`or `None`

Let us describe the input parameters only; outputs are similar. The parameter `inputs` specifies the inputs for the SUT. The value can be an integer or a list of strings. If a list of strings is given, say `["speed", "brake"]`, we have two inputs with names `"speed"` and `"brake"`. If a number is used, say `2`, then there are two inputs with default names `"i0"` and `"i1"` etc. The number of inputs is the number of signals for SUTs with signal inputs and the number of components for SUTs with vector inputs.

The parameter `input_range` specifies the ranges for the inputs as a list of 2-tuples of floats, for example `[[0, 100], [0, 325]]`. It is allowed to use `None` as an interval to indicate an unknown value. If the parameter `inputs` is omitted, it is inferred from this parameter. If `inputs` is specified, then intervals can be omitted and the value `None` is automatically used for missing values.

Neither `inputs` nor `input_range` needs to be specified in the `parameters` dictionary if they are set in the SUT class `__init__` method.

In addition to the above parameters, each SUT has attributes `idim` and `odim` available after `setup` has been called. These simply contain number of inputs (dimension of a vector or number of signals) and number of outputs (dimension of a vector or number of signals) respectively.

## Common SUT Methods

## Inputs and Outputs

## Input and Output Normalization

## Input Validity

## Exceptions
TODO

