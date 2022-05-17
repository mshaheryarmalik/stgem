import numpy as np

from stgem.objective import FalsifySTL

from util import build_specification

def get_falsifying_input_robustness(selected_specification):
    # The falsifying inputs are from
    # https://gitlab.com/gernst/ARCH-COMP/-/blob/FALS/2021/FALS/breach/breach_results.csv
    if selected_specification == "SC":
        raise Exception("We do not know any falsifying input for SC.")
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    A = np.fromstring(s[1:-1].replace(";", " "), sep=" ").reshape(-1, 2)
    timestamps = A[:,0]
    signals = A[:,1].reshape(1, -1)

    sut, specifications, scale, strict_horizon_check = build_specification(selected_specification)
    sut.setup(None, None)
    objectives = [FalsifySTL(specification=specification, scale=False, strict_horizon_check=strict_horizon_check) for specification in specifications]
    for objective in objectives:
        objective.setup(sut)

    sut_output = sut._execute_signal_signal(timestamps, signals)
    output = [objective(sut_output) for objective in objectives]
    
    return output

if __name__ == "__main__":
    specifications = ["SC"]

    # TODO: What are correct robustness values here?
    correct_robustness = {"SC": (0, 1, 6)}
    for specification in specifications:
        output = get_falsifying_input_robustness(specification)
        print(output)
        idx, robustness, precision = correct_robustness[specification]
        if abs(output[idx] - robustness) >= 10**(-precision):
            print("Incorrect output robustness {} for specification. Expected {}.".format(output[idx], robustness))
            raise SystemExit

    print("Correct!")

