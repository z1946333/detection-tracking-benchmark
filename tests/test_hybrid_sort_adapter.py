import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party' / 'HybridSORT'))
from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort


def make_args():
    return SimpleNamespace(
        track_thresh=0.30, TCM_first_step=False, TCM_byte_step=False,
        TCM_first_step_weight=1.0, TCM_byte_step_weight=1.0,
        low_thresh=0.10, use_byte=False,
    )


class HybridSortRegression(unittest.TestCase):
    def test_second_frame_uses_predict_scores(self):
        tracker = Hybrid_Sort(make_args(), det_thresh=0.30, max_age=30,
                              min_hits=1, iou_threshold=0.30, delta_t=3,
                              asso_func='iou', inertia=0.20, use_byte=False)
        det = np.array([[20, 20, 80, 100, 0.9]], dtype=np.float32)
        frame_shape = [120, 160]
        first = tracker.update(det, frame_shape, frame_shape)
        second = tracker.update(det, frame_shape, frame_shape)
        self.assertEqual(first.shape[1], 5)
        self.assertEqual(second.shape[1], 5)
        self.assertEqual(int(first[0, 4]), int(second[0, 4]))


if __name__ == '__main__':
    unittest.main()
