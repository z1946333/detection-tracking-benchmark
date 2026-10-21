import sys
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deep_ocsort_adapter import OSNetEmbedder, make_tracker


class DeepOCSortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.embedder = OSNetEmbedder(ROOT, ROOT / 'third_party/StrongSORT/osnet_x0_25_msmt17.pt', 'cpu')

    def test_real_features_are_finite_normalized_and_image_dependent(self):
        image = np.random.default_rng(42).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        boxes = np.array([[0, 0, 50, 100], [80, 10, 150, 110]])
        features = self.embedder.compute_embedding(image, boxes, 'test:1')
        self.assertEqual(features.shape, (2, 512))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)
        self.assertFalse(np.allclose(features[0], features[1]))

    def test_tracking_identity_and_empty_frames(self):
        tracker = make_tracker(ROOT, self.embedder)
        self.assertEqual(tracker.__class__.__module__, 'src.vendor.deep_ocsort.ocsort')
        self.assertFalse(tracker.aw_off)
        self.assertFalse(tracker.embedding_off)
        image = np.random.default_rng(1).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        shape = torch.empty(1, 3, 120, 160, device='meta')
        detections = np.array([[20, 20, 70, 100, .9]])
        first = tracker.update(detections.copy(), shape, image, 'unit:1')
        second = tracker.update(detections.copy(), shape, image, 'unit:2')
        self.assertEqual(first[0, 4], second[0, 4])
        tracker.update(np.empty((0, 5)), shape, image, 'unit:3')
        self.assertEqual(tracker.frame_count, 3)
        self.assertEqual(tracker.trackers[0].time_since_update, 1)

    def test_wrong_backbone_fails_loudly(self):
        state = dict(self.embedder.model.state_dict())
        state['conv1.conv.weight'] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            self.embedder.model.load_state_dict(state, strict=True)


if __name__ == '__main__':
    unittest.main()
