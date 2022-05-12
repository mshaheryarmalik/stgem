import tltk_mtl as STL

from stgem.sut.matlab.sut import Matlab
from stgem.objective import FalsifySTL

from f16_python_sut import F16GCAS_PYTHON2, F16GCAS_PYTHON3

def build_specification(selected_specification, asut=None):
    from math import pi

    # ARCH-COMP
    roll_range = [0.2*pi, 0.2833*pi]
    pitch_range = [-0.4*pi, -0.35*pi]
    yaw_range = [-0.375*pi, -0.125*pi]
    # PART-X
    """
    roll_range = [0.2*pi, 0.2833*pi]
    pitch_range = [-0.5*pi, -0.54*pi]
    yaw_range = [0.25*pi, 0.375*pi]
    """
    # FULL
    """
    roll_range = [-pi, pi]
    pitch_range = [-pi, pi]
    yaw_range = [-pi, pi]
    """

    sut_parameters = {"model_file": "problems/arch-comp-2021/f16/run_f16",
                      "init_model_file": "problems/arch-comp-2021/f16/init_f16",
                      "input_type": "vector",
                      "output_type": "signal",
                      "inputs": ["ROLL", "PITCH", "YAW"],
                      "outputs": ["ALTITUDE"],
                      "input_range": [roll_range, pitch_range, yaw_range],
                      "output_range": [[0, 4040]], # Starting altitude defined in init_f16.m.
                      "initial_altitude": 4040, # Used by the Python SUTs.
                      "simulation_time": 15
                     }

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab(sut_parameters)
        #asut = F16GCAS_PYTHON2(sut_parameters)
        #asut = F16GCAS_PYTHON3(sut_parameters)

    # Notice that here the input is a vector.

    scale = True
    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "F16":
        # always[0,15] ALTITUDE > 0
        specification = STL.Global(0, 15, FalsifySTL.StrictlyGreaterThan(1, 0, 0, 0, S("ALTITUDE")))

        specifications = [specification]
        strict_horizon_check = True
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return asut, specifications, scale, strict_horizon_check

