"""Deep OC-SORT association with an explicit OSNet-x0.25 ReID variant.

This is not DeepSORT and not the original paper's ReID configuration.
See docs/deep-ocsort-fix.md for provenance and evaluation limitations.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from .vendor.deep_ocsort.osnet import osnet_x0_25
from .vendor.deep_ocsort.ocsort import OCSort


class OSNetEmbedder:
    def __init__(self, root, weights, device='cuda:0'):
        self.device = torch.device(device)
        self.weights = Path(weights)
        checkpoint = torch.load(self.weights, map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint.get('net_dict', checkpoint))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        if 'classifier.weight' not in state:
            raise ValueError('Expected OSNet checkpoint with classifier.weight; refusing partial loading')
        # Match the checkpoint head (1041 for the supplied MSMT17 file), not pig classes.
        self.model = osnet_x0_25(num_classes=state['classifier.weight'].shape[0], pretrained=False)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
        self.calls = 0

    def compute_embedding(self, image, boxes, tag):
        if len(boxes) == 0:
            return np.empty((0, 512), dtype=np.float32)
        h, w = image.shape[:2]
        crops = []
        for box in boxes:
            x1, y1 = np.floor(box[:2]).astype(int)
            x2, y2 = np.ceil(box[2:]).astype(int)
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f'Invalid ReID crop {box} at {tag}')
            crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crops.append(cv2.resize(crop, (128, 256)).transpose(2, 0, 1))
        result = []
        with torch.inference_mode():
            for start in range(0, len(crops), 32):
                batch = torch.from_numpy(np.stack(crops[start:start+32])).to(self.device).float() / 255
                features = self.model((batch - self.mean) / self.std)
                result.append(torch.nn.functional.normalize(features, dim=1).cpu().numpy())
        self.calls += 1
        features = np.concatenate(result)
        if not np.isfinite(features).all():
            raise RuntimeError('Non-finite ReID features')
        return features

    def dump_cache(self):
        pass  # No disk cache: prevents reusing stale features for changed detections.


def make_tracker(root, embedder):
    return OCSort(det_thresh=.3, max_age=30, min_hits=3, iou_threshold=.3,
                  embedding_off=False, aw_off=False, cmc_off=False,
                  grid_off=True, new_kf_off=True, w_association_emb=.75,
                  aw_param=.5, embedder=embedder)
