import numpy as np

from stgem.objective import FalsifySTL

from util import build_specification

def get_falsifying_input_robustness(selected_specification):
    if selected_specification == "F16":
        test = [0.87718976, -1.2524469, -0.4304129]
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    sut, specifications, scale, strict_horizon_check = build_specification(selected_specification)
    sut.setup(None, None)
    objectives = [FalsifySTL(specification=specification, scale=False, strict_horizon_check=strict_horizon_check) for specification in specifications]
    for objective in objectives:
        objective.setup(sut)

    sut_output = sut._execute_vector_signal(test)
    output = [objective(sut_output) for objective in objectives]
    
    return output

if __name__ == "__main__":
    specifications = ["F16"]
    correct_robustness = {"F16": (0, -1.6128, 4)}
    for specification in specifications:
        output = get_falsifying_input_robustness(specification)
        idx, robustness, precision = correct_robustness[specification]
        if abs(output[idx] - robustness) >= 10**(-precision):
            print("Incorrect output robustness {} for specification. Expected {}.".format(output[idx], robustness))
            raise SystemExit

    print("Correct!")

