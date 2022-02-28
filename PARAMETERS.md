SUT Parameters
==============

A SUT can be configured through sut_parameters key in the job description file. These settings are mainly SUT-specific, but there are some general settings regarding the inputs and outputs. The following parameters exist:

* inputs
* input_range
* outputs
* output_range

Let us describe input configuration only; output is similar.

The parameter `inputs` specifies the inputs for the SUT. The value can be a list of strings or an integer. If a list of strings is given, say `["speed", "brake"]`, we have two inputs with names `"speed"` and `"brake"`. If a number is used, say `2`, then there are two inputs with default names `"i0"` and `"i1"` etc. The number of inputs is the number of signals for SUTs with signal inputs and the number of components for SUTs with vector inputs.

The parameter `input_range` specifies the ranges for the inputs as a list of 2-tuples of floats, for example `[[0, 100], [0, 325]]`. It is allowed to use `None` as an interval to indicate an unknown value. If the parameter `inputs` is omitted, it is inferred from this parameter. If `inputs` is specified, then intervals can be omitted and the value `None` is automatically used for missing values.

Neither `inputs` nor `input_range` needs to be specified if they are set in the used SUT class.

