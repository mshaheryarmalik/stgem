# SUT
A system under test (SUT) is represented by a class that inherits from the base class SUT. A SUT is a deterministic function that maps an input to an output. Both inputs and outputs can be either vectors or discrete signals. The execution of this function is achieved by the method `execute_test` which takes a SUTInput as an input and returns a SUTResult object.

## SUT Initialization
The `__init__` method of a SUT by convention takes as an argument a dictionary `parameters` which is supposed to contain all static (strings, integers, etc.) parameters the SUT needs. Other arguments can be specified, but we recommend minimalism here. The base class saves the input dictionary `parameters` as `x.parameters`. For convenience, we allow writing `x.foo` in place of `x.parameters["foo"]`, and it is important for this functionality that an inheriting class calls the parent class `__init__` method. Since some SUTs can be resource-heavy, we recommend to design SUT objects that are reusable. This means that they should not depend on external objective functions, algorithms, etc.

Each SUT has a `setup` method (taking no arguments) which is called in an STGEM object just before the generator is run. The purpose is to provide a two-step initialization where the user can alter the SUT object after it has been initialized but before it is being setup for use. We recommend that all resource-heavy initialization goes into the `setup` method. In this way several SUT objects can be initialized in advance without significant resource penalties. The `setup` method needs to be idempotent meaning that calling it several times results in the same outcome. It is mandatory to call the parent class `setup` method.

## Common SUT Attributes

## Common SUT Methods

## Inputs and Outputs

## Input and Output Normalization

## Input Validity

## Exceptions
TODO

