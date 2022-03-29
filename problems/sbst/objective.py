from stgem.objective.objective import Objective

class MaxOOB(Objective):
    """
    Objective which picks the maximum M from an output signal and returns 1-M
    for minimization.
    """

    def __init__(self, sut):
        super().__init__(sut)

        self.dim = 1

    def __call__(self, timestamps, signals):
        return 1 - max(signals[0])
