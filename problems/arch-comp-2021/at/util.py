import tltk_mtl as STL

from stgem.sut.matlab.sut import Matlab_Simulink
from stgem.objective import FalsifySTL

def build_specification(selected_specification, asut=None):
    """Builds a specification object and a SUT for the selected specification.
    In addition, returns if scaling and strict horizon check should be used for
    the specification. A previously created SUT can be passed as an argument,
    and then it will be reused."""

    sut_parameters = {"model_file": "problems/arch-comp-2021/at/Autotrans_shift",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["THROTTLE", "BRAKE"],
                      "outputs": ["SPEED", "RPM", "GEAR"],
                      "input_range": [[0, 100], [0, 325]],
                      "output_range": [[0, 121], [0, 4800], [0, 4]],
                      "simulation_time": 30,
                      #"time_slices": [30, 30],
                      "time_slices": [5, 5],
                      "sampling_step": 0.01
                     }

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab_Simulink(sut_parameters)

    # Some ARCH-COMP specifications have requirements whose horizon is longer than
    # the output signal for some reason. Thus strict horizon check needs to be
    # disabled in some cases.
    #
    # The requirements ATX1* and ATX2 are from "Falsification of hybrid systems
    # using adaptive probabilistic search" by Ernst et al.
    scale = True
    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "AT1":
        # always[0,20](SPEED < 120)
        specification = STL.Global(0, 20, FalsifySTL.StrictlyLessThan(1, 0, 0, 120, S("SPEED")))

        specifications = [specification]
        strict_horizon_check = True
    elif selected_specification == "AT2":
        # always[0,10](RPM < 4750)
        specification = STL.Global(0, 10, FalsifySTL.StrictlyLessThan(1, 0, 0, 4750, S("RPM")))

        specifications = [specification]
        strict_horizon_check = True
    elif selected_specification.startswith("AT5"):
        # This is modified from ARCH-COMP to include the next operator which is
        # available as we use discrete time STL.
        # always[0,30]( ( not(GEAR == {0}) and (eventually[0.001,0.1](GEAR == {0})) ) implies ( eventually[0.001,0.1]( always[0,2.5](GEAR == {0}) ) ) )"
        G = int(selected_specification[-1])
        # not(GEAR == {0}) and (eventually[0.001,0.1](GEAR == {0}))
        L = STL.And(STL.Not(STL.Equals(1, 0, 0, G, S("GEAR"))), STL.Next(STL.Equals(1, 0, 0, G, S("GEAR"))))
        # eventually[0.001,0.1]( always[0,2.5](GEAR == {0}) )
        R = STL.Next(STL.Global(0, 2.5, STL.Equals(1, 0, 0, G, S("GEAR"))))

        specification = STL.Global(0, 30, STL.Implication(L, R))

        specifications = [specification]
        strict_horizon_check = False
    elif selected_specification.startswith("AT6"):
        A = selected_specification[-1]

        def getSpecification(A):
            if A == "A":
                UB = 4
                SL = 35
            elif A == "B":
                UB = 8
                SL = 50
            else:
                UB = 20
                SL = 65
              
            # (always[0,30](RPM < 3000)) implies (always[0,{0}](SPEED < {1}))
            L = STL.Global(0, 30, FalsifySTL.StrictlyLessThan(1, 0, 0, 3000, S("RPM")))
            R = STL.Global(0, UB, FalsifySTL.StrictlyLessThan(1, 0, 0, SL, S("SPEED")))
            return STL.Implication(L, R)

        if selected_specification.endswith("ABC"):
            specification = STL.And(STL.And(getSpecification("A"), getSpecification("B")), getSpecification("C"))

            specifications = [getSpecification("A"), getSpecification("B"), getSpecification("C")]
            #specifications = [specification]
        else:
            specification = getSpecification(A)

            specifications = [specification]

        strict_horizon_check = True
    elif selected_specification.startswith("ATX1"):
        # always[0,30]( GEAR == {} implies SPEED > 10*{} )
        # Only for {} == 3 or {} == 4.
        G = int(selected_specification[-1])

        L = STL.Equals(1, 0, 0, G, S("GEAR"))
        R = FalsifySTL.StrictlyGreaterThan(1, 0, 0, 10*G, S("SPEED"))

        specification = STL.Global(0, 30, STL.Implication(L, R))

        specifications = [specification]
        strict_horizon_check = True
    elif selected_specification == "ATX2":
        # not(always[0,30]( 50 <= SPEED <= 60 ))
        L = FalsifySTL.GreaterThan(1, 0, 0, 50, S("SPEED"))
        R = STL.LessThan(1, 0, 0, 60, S("SPEED"))

        specification = STL.Not(STL.Global(0, 30, STL.And(L, R)))

        specifications = [specification]
        strict_horizon_check = True
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return asut, specifications, scale, strict_horizon_check

