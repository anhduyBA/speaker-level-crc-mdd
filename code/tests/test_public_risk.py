import unittest
from unittest.mock import patch

import numpy as np

from vpa.public_risk import (calibrate_crc, conformal_quantile, evaluation_bootstrap,
                             interval_evaluation, score_intervals, speaker_split,
                             valid_phone_mask, warning_metrics, warning_rows)


class PublicRiskTests(unittest.TestCase):
    def test_exact_rank_twenty_four_not_interpolated(self):
        q, rank = conformal_quantile(np.arange(25) / 12, .1)
        self.assertEqual(rank, 24)
        self.assertEqual(q, 23 / 12)

    def test_ties_remain_inclusive_at_quantile(self):
        q, rank = conformal_quantile(np.array([0.] * 22 + [.5, .5, 1.]), .1)
        self.assertEqual((q, rank), (.5, 24))
        result, _ = interval_evaluation(np.array([0., 1.]), np.array([.5, .5]), np.array(["A", "A"]), q)
        self.assertEqual(result["simultaneous_speaker_coverage"], 1.)

    def test_small_calibration_uses_known_bounded_full_interval(self):
        self.assertEqual(conformal_quantile(np.array([0., .1]), .1), (2., 3))
        lower, upper = score_intervals(np.array([0., .7, 2.]), 2.)
        np.testing.assert_array_equal(lower, 0.)
        np.testing.assert_array_equal(upper, 2.)

    def test_closed_interval_clips_score_boundaries(self):
        lower, upper = score_intervals(np.array([0., 1., 2.]), .5)
        np.testing.assert_array_equal(lower, [0., .5, 1.5])
        np.testing.assert_array_equal(upper, [.5, 1.5, 2.])

    def test_zero_gold_retained_padding_removed(self):
        labels = np.array([[[5, -1], [5, 0], [5, 1], [5, 2]]])
        np.testing.assert_array_equal(valid_phone_mask(labels), [[False, True, True, True]])
        for bad in (-.1, 2.1, np.nan):
            with self.assertRaises(ValueError):
                valid_phone_mask(np.array([[[0, bad]]]))

    def test_crc_strict_threshold_and_fixed_denominator(self):
        gold = np.array([2., 2., 0., 1.])
        priority = np.array([.2, .7, .2, .8])
        speakers = np.array(["A"] * 4)
        at_tie = warning_rows(gold, priority, speakers, .2)[0]
        self.assertEqual(at_tie["false_flags"], 1)
        self.assertEqual(at_tie["flags"], 2)
        self.assertEqual(at_tie["crc_loss"], .5)
        below = warning_rows(gold, priority, speakers, np.nextafter(.2, 0))[0]
        self.assertEqual(below["false_flags"], 2)
        self.assertEqual(below["crc_loss"], 1.)

    def test_no_gold2_speaker_retained_with_zero_loss(self):
        rows = warning_rows(np.array([0., 1., 2.]), np.ones(3), np.array(["A", "A", "B"]), .5)
        self.assertEqual(rows[0]["aggregate_score_2"], 0)
        self.assertEqual(rows[0]["crc_loss"], 0.)
        self.assertEqual(rows[0]["full_recall_score_below_2"], 1.)
        self.assertEqual(warning_metrics(rows)["speaker_macro_crc_loss"], .5)

    def test_crc_finite_sample_correction_and_least_feasible_tie(self):
        # 25 speakers, one eligible phone each, all priority .2. Above .2 removes all.
        result = calibrate_crc(np.full(25, 2.), np.full(25, .2), np.array([f"S{i:02}" for i in range(25)]))
        self.assertEqual(result["threshold"], .2)
        self.assertAlmostEqual(result["finite_sample_correction"], 1 / 26)
        self.assertAlmostEqual(result["corrected_empirical_risk"], 1 / 26)
        self.assertFalse(result["empty_feasible_set_fallback"])

    def test_crc_fallback_when_sample_too_small(self):
        result = calibrate_crc(np.array([2.]), np.array([.8]), np.array(["A"]), .1)
        self.assertEqual(result["threshold"], 1.)
        self.assertTrue(result["empty_feasible_set_fallback"])
        self.assertEqual(result["calibration_rows"][0]["false_flags"], 0)

    def test_crc_no_gold2_anywhere_is_defined(self):
        result = calibrate_crc(np.array([0., 1.]), np.array([.1, .8]), np.array(["A", "B"]), .5)
        self.assertEqual(result["threshold"], 0.)
        self.assertEqual(result["empirical_macro_loss"], 0.)

    def test_no_flags_precision_null_and_full_recall_zero(self):
        row = warning_metrics(warning_rows(np.array([2., 0.]), np.ones(2), np.array(["A", "A"]), 1.))
        self.assertEqual(row["flags"], 0)
        self.assertIsNone(row["false_fraction_among_flags_NOT_controlled"])
        self.assertIsNone(row["precision_score_derived_NOT_controlled"])
        self.assertEqual(row["full_recall_score_below_2"], 0.)

    def test_partition_deterministic_disjoint_complete_and_order_independent(self):
        ids = np.array([f"{i:04}" for i in range(125)])
        split = speaker_split(np.r_[ids, ids])
        self.assertEqual([len(split[k]) for k in ("fit", "calibration", "evaluation")], [75, 25, 25])
        self.assertEqual(split, speaker_split(ids[::-1]))
        groups = [set(split[k]) for k in ("fit", "calibration", "evaluation")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(set.union(*groups), set(ids))

    def test_one_speaker_with_one_bad_phone_loses_simultaneous_coverage(self):
        gold = np.array([2., 2., 2., 0.])
        pred = np.array([2., 2., 2., 1.])
        speakers = np.array(["A", "A", "B", "B"])
        summary, _ = interval_evaluation(gold, pred, speakers, .1)
        self.assertEqual(summary["pooled_phone_coverage"], .75)
        self.assertEqual(summary["simultaneous_speaker_coverage"], .5)

    def test_bootstrap_resamples_speakers_fixed_calibration(self):
        y, p, s = np.array([2., 0., 2., 1.]), np.array([2., 1., 1., 1.]), np.array(["A", "A", "B", "B"])
        _, intervals = interval_evaluation(y, p, s, 1.)
        warnings = warning_rows(y, 1 - p / 2, s, 1.)
        boot = evaluation_bootstrap(intervals, warnings, draws=50)
        self.assertEqual(boot["intervals"]["simultaneous_speaker_coverage"]["percentile95"], [1., 1.])
        self.assertEqual(boot["intervals"]["false_fraction_among_flags_NOT_controlled"]["n_defined_resamples"], 0)


if __name__ == "__main__":
    unittest.main()
