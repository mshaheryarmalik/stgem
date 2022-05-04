import tltk_mtl as STL

from stgem.sut.matlab.sut import Matlab
from stgem.objective import FalsifySTL

def build_specification(selected_specification, asut=None):
    """Builds a specification object and a SUT for the selected specification.
    In addition, returns if scaling and strict horizon check should be used for
    the specification. A previously created SUT can be passed as an argument,
    and then it will be reused."""

    # Notice that this only implements the Instance 2 version of the problem where
    # the input signal is split into exactly 20 segments.

    sut_parameters = {"model_file": "problems/arch-comp-2021/sc/run_steamcondenser",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["FS"],
                      "outputs": ["T", "FCW", "Q", "PRESSURE"],
                      "input_range": [[3.99, 4.01]],
                      "output_range": [None, None, None, [86, 90]],
                      "simulation_time": 35,
                      "time_slices": [1.75],
                      "sampling_step": 0.5
                     }

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab(sut_parameters)

    scale = True
    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "SC":
        # always[30,35](87 <= pressure <= 87.5)
        L = STL.LessThan(0, 87, 1, 0, None, S("PRESSURE"))
        R = STL.LessThan(1, 0, 0, 87.5, S("PRESSURE"))
        inequality = STL.And(L, R)
        specification = STL.Global(30, 35, inequality)

        specifications = [specification]
        strict_horizon_check = True
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return asut, specifications, scale, strict_horizon_check

