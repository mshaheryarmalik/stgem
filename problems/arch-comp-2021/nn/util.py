import tltk_mtl as STL

from stgem.sut.matlab.sut import Matlab
from stgem.objective import FalsifySTL

def build_specification(selected_specification, asut=None):
    """Builds a specification object and a SUT for the selected specification.
    In addition, returns if scaling and strict horizon check should be used for
    the specification. A previously created SUT can be passed as an argument,
    and then it will be reused."""

    # Notice that this only implements the Instance 2 version of the problem where
    # the input signal is split into exactly 3 segments.

    if selected_specification == "NN":
        ref_input_range = [1, 3]
    elif selected_specification == "NNX":
        ref_input_range = [1.95, 2.05]
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    sut_parameters = {"model_file": "problems/arch-comp-2021/nn/run_neural",
                      "init_model_file": "problems/arch-comp-2021/nn/init_neural",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["REF"],
                      "outputs": ["POS"],
                      "input_range": [ref_input_range],
                      "output_range": [[0, 4]],
                      "simulation_time": 40,
                      "time_slices": [13.33],
                      "sampling_step": 0.01
                     }

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab(sut_parameters)

    scale = True
    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "NN":
        alpha = 0.005
        beta = 0.03
        # inequality1 := |POS - REF| > alpha + beta*|REF|
        # inequality2 := alpha + beta*|REF| > |POS - REF|
        inequality1 = FalsifySTL.StrictlyGreaterThan(1, 0, beta, alpha, STL.Abs(STL.Subtract(S("POS"), S("REF"))), STL.Abs(S("REF")))
        inequality2 = FalsifySTL.StrictlyGreaterThan(beta, alpha, 1, 0, STL.Abs(S("REF")), STL.Abs(STL.Subtract(S("POS"), S("REF"))))
        # always[1,37]( inequality implies (always[0,2]( eventually[0,1] not inequality )) )
        specification = STL.Global(1, 37, STL.Implication(inequality1, STL.Finally(0, 2, STL.Global(0, 1, inequality2))))

        specifications = [specification]
        strict_horizon_check = True
    elif selected_specification == "NNX":
        # eventually[0,1](POS > 3.2)
        F1 = STL.Finally(0, 1, FalsifySTL.StrictlyGreaterThan(1, 0, 0, 3.2, S("POS")))
        # eventually[1,1.5]( always[0,0.5](1.75 < POS < 2.25) )
        L = FalsifySTL.StrictlyLessThan(0, 1.75, 1, 0, None, S("POS"))
        R = FalsifySTL.StrictlyLessThan(1, 0, 0, 2.25, S("POS"))
        inequality = STL.And(L, R)
        F2 = STL.Finally(1, 1.5, STL.Global(0, 0.5, inequality))
        # always[2,3](1.825 < POS < 2.175)
        L = FalsifySTL.StrictlyLessThan(0, 1.825, 1, 0, None, S("POS"))
        R = FalsifySTL.StrictlyLessThan(1, 0, 0, 2.175, S("POS"))
        inequality = STL.And(L, R)
        F3 = STL.Global(2, 3, inequality)

        conjunctive_specification = STL.And(F1, STL.And(F2, F3))

        #specifications = [conjunctive_specification]
        specifications = [F1, F2, F3]
        strict_horizon_check = True
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return asut, specifications, scale, strict_horizon_check

