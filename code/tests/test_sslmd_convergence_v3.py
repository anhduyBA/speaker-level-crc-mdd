import unittest
import numpy as np
from vpa.sslmd_convergence_v3 import make_head,pick_capped


class ConvergenceContractTests(unittest.TestCase):
    def test_uniform_is_layer_mean_with_same_linear_init(self):
        import torch
        torch.manual_seed(7);uniform=make_head('uniform',8)
        torch.manual_seed(7);last=make_head('last',8)
        self.assertTrue(torch.equal(uniform.linear.weight,last.linear.weight))
        layers=torch.randn(12,5,8)
        self.assertTrue(torch.allclose(uniform(layers),last.linear(layers.mean(0))))
        self.assertEqual(sum(x.numel() for x in uniform.parameters()),sum(x.numel() for x in last.parameters()))

    def test_mixed_at_zero_init_equals_uniform(self):
        import torch
        torch.manual_seed(3);mixed=make_head('mixed',8)
        torch.manual_seed(3);uniform=make_head('uniform',8)
        layers=torch.randn(12,4,8)
        self.assertTrue(torch.allclose(mixed(layers),uniform(layers),atol=1e-6))

    def test_capped_selection_ignores_later_epochs(self):
        h=[{'calibration_pauc':v} for v in (.5,.7,.6,.9)]
        self.assertEqual(pick_capped(h,3),1)
        self.assertEqual(pick_capped(h,4),3)

    def test_capped_selection_keeps_earliest_tie(self):
        h=[{'calibration_pauc':v} for v in (.4,.8,.8,.8)]
        self.assertEqual(pick_capped(h,4),1)

    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError):make_head('weighted')

    def test_uniform_rejects_wrong_layer_count(self):
        import torch
        with self.assertRaises(ValueError):make_head('uniform',8)(torch.randn(11,3,8))


if __name__=='__main__':
    unittest.main()
