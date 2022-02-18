import os

import numpy as np

from objective.objective import Objective

class AT_Objective_1(Objective):
    """
    Implements the first requirement and its robustness functions from the
    ITEQS paper as an objective function.
    """

    def __init__(self, sut):
        super().__init__(sut)
        self.dim = 1

    def __call__(self, timestamps, signals):
        T = timestamps
        S = signals

        SPEED = 0 # first signal
        RPM = 1   # second signal
        GEAR = 2  # third signal (not used)

        # Requirement 1
        # (always[0,30] (RPM < 3000)) -> (always[0,4] (SPEED < 35))
        M_RPM = max(signals[RPM])
        M_SPEED = max(x for n, x in enumerate(signals[SPEED]) if timestamps[n] <= 4.0)
        if M_RPM < 3000:
            R1 = 0.5*(35 - M_SPEED)/35
        else:
            R1 = M_RPM/3000 - 0.5

        R1 = max(0, min(1, R1))

        return R1

class AT_Objective_2(Objective):
    """
    Implements the second requirement and its robustness functions from the
    ITEQS paper as an objective function.
    """

    def __init__(self, sut):
        super().__init__(sut)
        self.dim = 1

    def __call__(self, timestamps, signals):
        T = timestamps
        S = signals

        SPEED = 0 # first signal
        RPM = 1   # second signal
        GEAR = 2  # third signal (not used)

        # Requirement 2
        # (always[0,30] (RPM < 3000)) -> (always[0,8] (SPEED < 50))
        M_RPM = max(signals[RPM])
        M_SPEED = max(x for n, x in enumerate(signals[SPEED]) if timestamps[n] <= 8.0)
        if M_RPM < 3000:
            R2 = 0.5*(50 - M_SPEED)/50
        else:
            R2 = M_RPM/3000 - 0.5

        R2 = max(0, min(1, R2))

        return R2

class AT_Objective_3(Objective):
    """
    Implements the third requirement and its robustness functions from the
    ITEQS paper as an objective function.
    """

    def __init__(self, sut):
        super().__init__(sut)
        self.dim = 1

    def __call__(self, timestamps, signals):
        T = timestamps
        S = signals

        SPEED = 0 # first signal
        RPM = 1   # second signal
        GEAR = 2  # third signal (not used)

        # Requirement 3
        # Same as Requirement 1, but change [0,4] -> [0,20] and 35 -> 65.
        M_RPM = max(signals[RPM])
        M_SPEED = max(x for n, x in enumerate(signals[SPEED]) if timestamps[n] <= 20.0)
        if M_RPM < 3000:
            R3 = 0.5*(65 - M_SPEED)/65
        else:
            R3 = M_RPM/3000 - 0.5

        R3 = max(0, min(1, R3))

        return R3

