import unittest
import numpy as np
from vpa.sslmd_smoke import frame_map, pool_logits


class MappingTests(unittest.TestCase):
    def test_unknown_and_silence_are_not_negatives(self):
        m = frame_map([0, 1, 2], [0, -1, 1], [[1, 3], [4, 6], [6, 8]], 9)
        self.assertEqual(m['valid'].tolist(), [False, True, True, False, False, False, True, True, False])
        self.assertEqual(m['target'].tolist(), [-1, 0, 0, -1, -1, -1, 1, 1, -1])

    def test_overlaps_and_invalid_spans_rejected(self):
        for ranges in ([[1, 4], [3, 5]], [[-1, 2], [2, 3]], [[2, 1], [3, 4]], [[0, 2], [3, 10]]):
            with self.assertRaises(ValueError):
                frame_map([0, 1], [0, 1], ranges, 9)

    def test_zero_frame_position_stays_unsupported(self):
        m = frame_map([0, 1], [1, 0], [[0, 0], [1, 2]], 3)
        self.assertEqual(m['valid'].sum(), 1)
        self.assertEqual(pool_logits(np.zeros((3, 39)), [0, 1], [[0, 0], [1, 2]]), [None, 0.])

    def test_repeated_phone_has_distinct_position_owners(self):
        m = frame_map([2, 2], [0, 1], [[0, 2], [3, 5]], 5)
        self.assertEqual(m['owner'].tolist(), [0, 0, -1, 1, 1])
        self.assertEqual(m['target'].tolist(), [0, 0, -1, 1, 1])

    def test_pool_selects_phone_output_and_averages_logits(self):
        logits = np.zeros((4, 39)); logits[:, 2] = [-4, 2, 6, 12]
        self.assertEqual(pool_logits(logits, [2, 0], [[0, 2], [2, 4]]), [-1., 0.])

    def test_unknown_label_does_not_change_ownership(self):
        a = frame_map([0, 1], [-1, -1], [[0, 2], [2, 4]], 4)
        b = frame_map([0, 1], [0, 1], [[0, 2], [2, 4]], 4)
        np.testing.assert_array_equal(a['owner'], b['owner'])
        np.testing.assert_array_equal(a['phone'], b['phone'])
        self.assertFalse(a['valid'].any())


if __name__ == '__main__':
    unittest.main()
