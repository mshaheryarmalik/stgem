import numpy as np

# rtamt may have dependency problems. We continue even if we cannot import it
try:
    import rtamt
except:
    print("Cannot import rtamt. Objectives using rtamt will throw an exception.")
    import traceback
    traceback.print_exc()


class Objective:

    def __init__(self, sut):
        self.sut = sut

    def __call__(self, output):
        return output


class MaxOOB(Objective):
    """
    Objective function for a SUT with fixed-length vector outputs which selects
    the maximum among the specified components.
    """

    def __init__(self, sut, selected=None, scale=False, invert=False):
        super().__init__(sut)
        if not (isinstance(selected, list) or isinstance(selected, tuple) or selected is None):
            raise Exception("The parameter 'selected' must be None or a list or a tuple.")

        self.dim = 1
        self.selected = selected
        self.scale = scale
        self.invert = invert

    def __call__(self, output):
        return max(output)
