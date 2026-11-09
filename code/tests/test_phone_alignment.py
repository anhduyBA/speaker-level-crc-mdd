import itertools
import unittest

import numpy as np

from vpa.phone_alignment import PROBABILITY_FLOOR, posterior_align


class PhoneAlignmentTests(unittest.TestCase):
    def test_dynamic_program_matches_brute_force_maximum(self):
        p = np.array([[.7,.2,.1], [.1,.8,.1], [.2,.5,.3], [.1,.1,.8], [.6,.1,.3]])
        n_states = 5
        legal = []
        state_ids = np.array([0,1,0,2,0])
        for path in itertools.product(range(n_states), repeat=len(p)):
            if path[0] not in (0,1) or path[-1] not in (3,4):
                continue
            if any(b-a not in (0,1,2) or (b-a==2 and not (a%2==1 and b%2==1))
                   for a,b in zip(path,path[1:])):
                continue
            if not all(s in path for s in (1,3)):
                continue
            legal.append(sum(np.log(max(p[t,state_ids[s]],PROBABILITY_FLOOR)) for t,s in enumerate(path)))
        result = posterior_align(p,[1,2])
        self.assertAlmostEqual(result['best_score'],max(legal),places=12)
        self.assertEqual(result['phone_frame_ranges'],[[1,3],[3,4]])
        self.assertEqual(result['n_silence_frames'],2)

    def test_adjacent_repeated_phones_remain_distinct_mandatory_positions(self):
        result = posterior_align(np.array([[.1,.9],[.1,.9]]),[1,1])
        self.assertEqual(result['phone_frame_ranges'],[[0,1],[1,2]])
        self.assertEqual(result['state_path'].tolist(),[1,3])

    def test_mandatory_low_score_phone_cannot_be_skipped(self):
        p = np.array([[.01,.98,.01],[.01,.98,.01],[.01,.98,.01]])
        result = posterior_align(p,[1,2,1])
        self.assertEqual(result['phone_frame_ranges'],[[0,1],[1,2],[2,3]])
        self.assertEqual(result['state_path'].tolist(),[1,3,5])

    def test_leading_trailing_and_internal_silence_are_optional(self):
        p = np.array([[.01,.98,.01],[.98,.01,.01],[.01,.01,.98]])
        result = posterior_align(p,[1,2])
        self.assertEqual(result['state_path'].tolist(),[1,2,3])
        self.assertEqual(result['phone_frame_ranges'],[[0,1],[2,3]])
        self.assertEqual(posterior_align(np.array([[.01,.98,.01],[.01,.01,.98]]),[1,2])['state_path'].tolist(),[1,3])

    def test_equal_scores_have_deterministic_declared_ties(self):
        p = np.full((4,3),1/3)
        a = posterior_align(p,[1,2]);b = posterior_align(p,[1,2])
        np.testing.assert_array_equal(a['state_path'],b['state_path'])
        self.assertEqual(a['state_path'].tolist(),[1,3,3,3])

    def test_rejects_oov_silence_bad_ids_and_insufficient_frames(self):
        p=np.full((2,3),1/3)
        for ids in ([3],[-1],[0],[True],[1.0],[]):
            with self.assertRaises(ValueError):posterior_align(p,ids)
        with self.assertRaises(ValueError):posterior_align(p,[1,2,1])
        with self.assertRaises(ValueError):posterior_align(p,[1],silence_id=True)

    def test_rejects_invalid_posteriors_and_does_not_mutate_inputs(self):
        for p in (np.zeros((2,3)),np.array([[.5,np.nan,.5]]),np.array([[-.1,.6,.5]]),np.empty((0,3)),np.ones((2,1))):
            with self.assertRaises(ValueError):posterior_align(p,[1])
        p=np.array([[0.,1.,0.],[.2,.8,0.]])
        before=p.copy();result=posterior_align(p,[1])
        np.testing.assert_array_equal(before,p)
        self.assertTrue(np.isfinite(result['best_score']))


if __name__=='__main__':unittest.main()
