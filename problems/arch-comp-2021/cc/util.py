import tltk_mtl as STL

from stgem.sut.matlab.sut import Matlab_Simulink
from stgem.objective import FalsifySTL

def build_specification(selected_specification, asut=None):
    """Builds a specification object and a SUT for the selected specification.
    In addition, returns if scaling and strict horizon check should be used for
    the specification. A previously created SUT can be passed as an argument,
    and then it will be reused."""

    sut_parameters = {"model_file": "problems/arch-comp-2021/cc/cars",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["THROTTLE", "BRAKE"],
                      "outputs": ["Y1", "Y2", "Y3", "Y4", "Y5"],
                      "input_range": [[0, 1], [0, 1]],
                      "output_range": [[-5000, 0], [-5000, 0], [-5000, 0], [-5000, 0], [-5000, 0]],
                      "simulation_time": 100,
                      "time_slices": [5, 5],
                      "sampling_step": 0.5
                     }
    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab_Simulink(sut_parameters)

    # Some ARCH-COMP specifications have requirements whose horizon is longer than
    # the output signal for some reason. Thus strict horizon check needs to be
    # disabled in some cases.
    scale = True
    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "CC1":
        # always[0,100]( y5 - y4 <= 40 )
        specification = STL.Global(0, 100, STL.LessThan(1, 0, 1, 40, S("Y5"), S("Y4")))
        
        strict_horizon_check = True
        specifications = [specification]
    elif selected_specification == "CC2":
        # always[0,70]( eventually[0,30]( y5 - y4 >= 15 ) )
        specification = STL.Global(0, 70, STL.Finally(0, 30, FalsifySTL.GreaterThan(1, 0, 1, 15, S("Y5"), S("Y4"))))

        strict_horizon_check = True
        specifications = [specification]
    elif selected_specification == "CC3":
        # always[0,80]( (always[0,20]( y2 - y1 <= 20 )) or (eventually[0,20]( y5 - y4 >= 40 )) )
        L = STL.Global(0, 20, STL.LessThan(1, 0, 1, 20, S("Y2"), S("Y1"))) 
        R = STL.Finally(0, 20, FalsifySTL.GreaterThan(1, 0, 1, 40, S("Y5"), S("Y4")))
        specification = STL.Global(0, 80, STL.And(L, R))

        strict_horizon_check = True
        specifications = [specification]
    elif selected_specification == "CC4":
        # always[0,65]( eventually[0,30]( always[0,20]( y5 - y4 >= 8 ) ) )
        specification = STL.Global(0, 65, STL.Finally(0, 30, STL.Global(0, 20, FalsifySTL.GreaterThan(1, 0, 1, 8, S("Y5"), S("Y4")))))

        strict_horizon_check = False
        specifications = [specification]
    elif selected_specification == "CC5":
        # always[0,72]( eventually[0,8]( always[0,5]( y2 - y1 >= 9 ) implies always[5,20]( y5 - y4 >= 9 ) ) )
        L = STL.Global(0, 5, FalsifySTL.GreaterThan(1, 0, 1, 9, S("Y2"), S("Y1")))
        R = STL.Global(5, 20, FalsifySTL.GreaterThan(1, 0, 1, 9, S("Y5"), S("Y4")))
        specification = STL.Global(0, 72, STL.Finally(0, 8, STL.Implication(L, R)))

        strict_horizon_check = True
        specifications = [specification]
    elif selected_specification == "CCX":
        # always[0,50]( y2 - y1 > 7.5 ) and always[0,50]( y3 - y2 > 7.5 ) and always[0,50]( y4 - y3 > 7.5 ) and always[0,50]( y5 - y4 > 7.5 )
        def getSpecification(N):
            return STL.Global(0, 50, FalsifySTL.StrictlyGreaterThan(1, 0, 1, 7.5, S("Y{}".format(N+1)), S("Y{}".format(N))))

        F1 = getSpecification(1)
        F2 = getSpecification(2)
        F3 = getSpecification(3)
        F4 = getSpecification(4)
        specification = STL.And(F1, STL.And(F2, STL.And(F3, F4) ) )

        strict_horizon_check = True
        #specifications = [specification]
        specifications = [F1, F2, F3, F4]
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return asut, specifications, scale, strict_horizon_check

