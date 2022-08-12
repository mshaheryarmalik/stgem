from parser.parser import parse
from _parser import ParserTestCase
import stgem.objective.Robustness as rbst

class SignalTestCase(ParserTestCase):
    def _do_test(self, phi: str, expected) -> None:

        signal = parse(phi, self._vars)

        assert isinstance(signal, rbst.Signal)

        self.assertEqual(signal.name, expected.name)
        self.assertEqual(signal.var_range, expected.var_range)

    def test_signal_pos_ns_less_than_pos(self) -> None:
        self._do_test(r"pred4 <= 1.0", self._preds["pred4"])

    def test_signal_pos_ns_less_than_neg(self) -> None:
        self._do_test(r"pred5 <= -1.0", self._preds["pred5"])

    def test_signal_neg_ns_less_than_pos(self) -> None:
        self._do_test(r"-pred6 <= 1.0", self._preds["pred6"])

    def test_signal_neg_ns_less_than_neg(self) -> None:
        self._do_test(r"-pred7 <= -1.0", self._preds["pred7"])

    def test_signal_pos_ns_greater_than_pos(self) -> None:
        self._do_test(r"pred7 >= 1.0", self._preds["pred7"])

    def test_signal_pos_ns_greater_than_neg(self) -> None:
        self._do_test(r"pred6 >= -1.0", self._preds["pred6"])

    def test_signal_neg_ns_greater_than_pos(self) -> None:
        self._do_test(r"-pred5 >= 1.0", self._preds["pred5"])

    def test_signal_neg_ns_greater_than_neg(self) -> None:
        self._do_test(r"-pred4 >= -1.0", self._preds["pred4"])

    def test_signal_pos_ns_less_than_pos_scientific_pos(self) -> None:
        self._do_test(r"pred8 <= 1.1e5", self._preds["pred8"])

    def test_signal_pos_ns_less_than_pos_scientific_neg(self) -> None:
        self._do_test(r"pred9 <= 1.12e-5", self._preds["pred9"])

    #def test_signal_difference_less_than_value(self) -> None:
        #self._do_test(r"pred8 - pred9 <= 1.12e-5", self._preds["pred9"])


