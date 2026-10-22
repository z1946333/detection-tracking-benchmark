"""Measure detector parameters, latency and peak GPU memory consistently.

Protocol: first N test images, batch=1, five warm-up calls, thirty timed calls
per model, CUDA synchronization around the model call, no disk I/O in timing.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF


ROOT = Path(r"D:\GitHub\detection-tracking-benchmark")
DATA = Path(r"D:\Datasets\processed_detection\images\test")
OUT = ROOT / "evaluation" / "detection"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def load_torchvision(name: str):
    from torchvision.models.detection.ssd import SSDClassificationHead, SSDRegressionHead
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    from torchvision.models.detection import (
        ssd300_vgg16,
        retinanet_resnet50_fpn,
        fasterrcnn_resnet50_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if name == "Faster R-CNN":
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        path = ROOT / "experiments/detection/fasterrcnn/best.pt"
    elif name == "RetinaNet":
        model = retinanet_resnet50_fpn(weights=None)
        n = model.anchor_generator.num_anchors_per_location()[0]
        model.head.classification_head = RetinaNetClassificationHead(
            model.backbone.out_channels, n, 2
        )
        path = ROOT / "experiments/detection/retinanet/best.pt"
    else:
        model = ssd300_vgg16(weights=None)
        channels = [512, 1024, 512, 256, 256, 256]
        anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDClassificationHead(channels, anchors, 2)
        model.head.regression_head = SSDRegressionHead(channels, anchors)
        path = ROOT / "experiments/detection/ssd300/best.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(DEVICE).eval(), "native torchvision input"


def load_yolov5():
    import torch as _torch

    path = ROOT / "experiments/detection/yolov5n3/weights/best.pt"
    model = _torch.hub.load(
        str(ROOT / "third_party/yolov5"), "custom", path=str(path), source="local"
    )
    model.conf = 0.001
    model.eval()
    return model, "640 px"


def load_yolov8():
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "experiments/detection/yolov8n/weights/best.pt"))
    model.model.to(DEVICE).eval()
    return model, "640 px"


def load_yolox():
    # The YOLOX checkpoint was serialized with numpy._core; map that module
    # name to the installed numpy.core without changing the environment.
    import pickle

    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            return super().find_class(module.replace("numpy._core", "numpy.core"), name)

    class CompatPickle:
        Unpickler = CompatUnpickler
        load = pickle.load
        loads = pickle.loads

    sys.path.insert(0, str(ROOT / "third_party/YOLOX"))
    from yolox.exp import get_exp

    exp = get_exp(str(ROOT / "configs/detection/yolox_pig.py"), None)
    model = exp.get_model().to(DEVICE).eval()
    ckpt = torch.load(
        ROOT / "YOLOX_outputs/yolox_pig/best_ckpt.pth",
        map_location="cpu",
        pickle_module=CompatPickle,
    )
    model.load_state_dict(ckpt["model"])
    return (model, exp), "640 px"


def make_inputs(paths, kind):
    items = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise FileNotFoundError(p)
        if kind == "torchvision":
            items.append(TF.to_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).to(DEVICE))
        else:
            items.append(bgr)
    return items


def infer_fn(name, model, item):
    if name in {"Faster R-CNN", "RetinaNet", "SSD300"}:
        with torch.inference_mode():
            return model([item])
    if name == "YOLOv5n":
        with torch.inference_mode():
            return model(item, size=640)
    if name == "YOLOv8n":
        return model.predict(source=item, imgsz=640, conf=0.001, device=0, verbose=False)
    yolo, exp = model
    from yolox.data.data_augment import ValTransform

    preproc = ValTransform(legacy=False)
    img, _ = preproc(item, None, exp.test_size)
    tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        return yolo(tensor)


def measure(name, loader, input_kind, paths, warmup, iterations):
    model, input_size = loader()
    if name == "YOLOv8n":
        param_count = params_m(model.model)
    elif name == "YOLOX":
        param_count = params_m(model[0])
    elif name == "YOLOv5n":
        param_count = params_m(model.model)
    else:
        param_count = params_m(model)
    items = make_inputs(paths, input_kind)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)
    for i in range(warmup):
        infer_fn(name, model, items[i % len(items)])
    sync()
    timings = []
    for i in range(iterations):
        item = items[i % len(items)]
        sync()
        start = time.perf_counter()
        infer_fn(name, model, item)
        sync()
        timings.append((time.perf_counter() - start) * 1000)
    peak = (torch.cuda.max_memory_allocated(DEVICE) / 1024**2) if DEVICE.type == "cuda" else None
    result = {
        "algorithm": name,
        "parameters_M": round(param_count, 3),
        "latency_mean_ms": round(float(np.mean(timings)), 3),
        "latency_median_ms": round(float(np.median(timings)), 3),
        "peak_gpu_memory_MB": round(float(peak), 3) if peak is not None else None,
        "fps_from_mean": round(1000.0 / float(np.mean(timings)), 3),
        "input_setting": input_size,
        "device": str(DEVICE),
        "images_used": len(paths),
        "warmup_calls": warmup,
        "timed_calls": iterations,
    }
    del items, model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=30)
    args = ap.parse_args()
    paths = sorted(DATA.glob("*.jpg"))[: args.images]
    if len(paths) < args.images:
        raise RuntimeError(f"Only {len(paths)} test images found, requested {args.images}")
    jobs = [
        ("Faster R-CNN", lambda: load_torchvision("Faster R-CNN"), "torchvision"),
        ("RetinaNet", lambda: load_torchvision("RetinaNet"), "torchvision"),
        ("SSD300", lambda: load_torchvision("SSD300"), "torchvision"),
        ("YOLOv5n", load_yolov5, "raw"),
        ("YOLOX", load_yolox, "raw"),
        ("YOLOv8n", load_yolov8, "raw"),
    ]
    results = []
    for name, loader, kind in jobs:
        print(f"========== {name} ==========")
        results.append(measure(name, loader, kind, paths, args.warmup, args.iterations))
        print(results[-1])
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "test_images": args.images,
            "warmup_calls": args.warmup,
            "timed_calls": args.iterations,
            "batch": 1,
            "timing_boundary": "model call only, synchronized CUDA, no disk I/O",
            "device": str(DEVICE),
        },
        "results": results,
    }
    (OUT / "efficiency_benchmark_20261022.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["algorithm,parameters_M,latency_mean_ms,latency_median_ms,peak_gpu_memory_MB,fps_from_mean,input_setting,device,images_used,warmup_calls,timed_calls"]
    for r in results:
        lines.append(",".join(str(r[k]) for k in ["algorithm","parameters_M","latency_mean_ms","latency_median_ms","peak_gpu_memory_MB","fps_from_mean","input_setting","device","images_used","warmup_calls","timed_calls"]))
    (OUT / "efficiency_benchmark_20261022.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {OUT / 'efficiency_benchmark_20261022.csv'}")


if __name__ == "__main__":
    main()
