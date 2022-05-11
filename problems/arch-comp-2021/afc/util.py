import tltk_mtl as STL

from stgem.sut.matlab.sut import Matlab
from stgem.objective import FalsifySTL

def build_specification(selected_specification, afc_mode, asut=None):
    """Builds a specification object and a SUT for the selected specification.
    In addition, returns if scaling and strict horizon check should be used for
    the specification. A previously created SUT can be passed as an argument,
    and then it will be reused."""

    afc_mode = "normal" # normal/power
    if afc_mode == "normal":
        throttle_range = [0, 61.2]
    elif afc_mode == "power":
        throttle_range = [61.2, 81.2]

    # Notice that the output MODE is never used in the requirements.
    sut_parameters = {"model_file": "problems/arch-comp-2021/afc/run_powertrain",
                      "init_model_file": "problems/arch-comp-2021/afc/init_powertrain",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["THROTTLE", "ENGINE"],
                      "outputs": ["MU", "MODE"],
                      "input_range": [throttle_range, [900, 1100]],
                      "output_range": [[-1, 1], [0, 1]],
                      "simulation_time": 50,
                      "time_slices": [5, 50],
                      "sampling_step": 0.01
                     }

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab(sut_parameters)

    # Some ARCH-COMP specifications have requirements whose horizon is longer than
    # the output signal for some reason. Thus strict horizon check needs to be
    # disabled in some cases.
    scale = True
    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "AFC27":
        beta = 0.008
        # rise := (THROTTLE < 8.8) and (eventually[0,0.05](THROTTLE > 40.0))
        L = FalsifySTL.StrictlyLessThan(1, 0, 0, 8.8, S("THROTTLE"))
        R = STL.Finally(0, 0.05, FalsifySTL.StrictlyGreaterThan(1, 0, 0, 40, S("THROTTLE")))
        rise = STL.And(L, R)
        # fall := (THROTTLE > 40.0) and (eventually[0,0.05](THROTTLE < 8.8))
        L = FalsifySTL.StrictlyGreaterThan(1, 0, 0, 40, S("THROTTLE"))
        R = STL.Finally(0, 0.05, FalsifySTL.StrictlyLessThan(1, 0, 0, 8.8, S("THROTTLE")))
        fall = STL.And(L, R)
        # consequence := always[1,5](abs(MU) < beta)
        consequence = STL.Global(1, 5, FalsifySTL.StrictlyLessThan(1, 0, 0, beta, STL.Abs(S("MU"))))
        # always[11,50]( (rise or fall) implies (consequence)
        specification = STL.Global(11, 50, STL.Implication(STL.Or(rise, fall), consequence))

        specifications = [specification]
        strict_horizon_check = False
    elif selected_specification == "AFC29":
        gamma = 0.007
        # always[11,50]( abs(MU) < gamma )
        specification = STL.Global(11, 50, FalsifySTL.StrictlyLessThan(1, 0, 0, gamma, STL.Abs(S("MU"))))

        specifications = [specification]
        strict_horizon_check = True
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return asut, specifications, scale, strict_horizon_check

