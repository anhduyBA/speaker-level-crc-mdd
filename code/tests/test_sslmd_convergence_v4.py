import unittest
import numpy as np
from vpa.sslmd_convergence_v4 import should_stop,holm,bootstrap_p


def h(values):return [{'calibration_pauc':v} for v in values]


class ConvergenceV4Tests(unittest.TestCase):
    def test_stops_after_patience_without_strict_gain(self):
        self.assertFalse(should_stop(h([.5,.6]+[.6]*19),20))
        self.assertTrue(should_stop(h([.5,.6]+[.6]*20),20))

    def test_strict_gain_resets_patience(self):
        self.assertFalse(should_stop(h([.6]*15+[.61]+[.6]*19),20))

    def test_empty_history_does_not_stop(self):
        self.assertFalse(should_stop([],20))

    def test_holm_matches_hand_computation(self):
        self.assertEqual([round(x,6) for x in holm([.01,.04,.03])],[.03,.06,.06])

    def test_holm_is_capped_at_one(self):
        self.assertEqual(holm([.6,.9]),[1.,1.])

    def test_bootstrap_p_extremes(self):
        self.assertAlmostEqual(bootstrap_p(np.ones(999)),2/1000)
        self.assertEqual(bootstrap_p(np.r_[np.ones(500),-np.ones(500)]),1.)


if __name__=='__main__':
    unittest.main()
