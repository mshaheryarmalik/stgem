from stgem.algorithm.random.model import Halton
from stgem.sut import SearchSpace
def test_halton():
    sp = SearchSpace()
    sp.input_dimension = 1
    sp.output_dimension = 1
    hal = Halton(parameters = {"size": 100}, search_space = sp)
    for _ in range(10):
        print(hal.generate_test())
test_halton()