"""Run an auditable ByteTrack benchmark on the PIG MOT sequences.

The detector and evaluation protocol intentionally match the existing tracker
experiments. Ultralytics writes normalized tracking labels; this script converts
them to ten-column MOTChallenge files and records a manifest for provenance.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(r"D:\GitHub\detection-tracking-benchmark")
DATA_ROOT = Path(r"D:\Datasets\dataset_B\PIG_MOT_TEST_DATA")
MODEL_PATH = ROOT / "experiments" / "detection" / "yolov8n" / "weights" / "best.pt"
RUN_NAME = "bytetrack_20261021"
EVAL_ROOT = ROOT / "evaluation" / "tracking" / RUN_NAME
TRACK_EVAL_ROOT = (
    ROOT / "third_party" / "TrackEval" / "data" / "trackers" /
    "mot_challenge" / "PIG-train" / "my_bytetrack"
)


def convert_labels(label_dir: Path, image_dir: Path, output_file: Path, max_frames: int | None) -> dict:
    image_paths = sorted(image_dir.glob("*.jpg"))
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise FileNotFoundError(f"No JPG frames found: {image_dir}")

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read image: {image_paths[0]}")
    height, width = first.shape[:2]

    lines: list[str] = []
    frames_with_tracks = set()
    for frame_id, image_path in enumerate(image_paths, start=1):
        label_file = label_dir / f"{image_path.stem}.txt"
        if not label_file.exists():
            continue
        for raw in label_file.read_text(encoding="utf-8").splitlines():
            parts = raw.split()
            # class, x_center, y_center, width, height, track_id
            if len(parts) < 6:
                continue
            _, x_center, y_center, box_w, box_h, track_id = map(float, parts[:6])
            x = (x_center - box_w / 2.0) * width
            y = (y_center - box_h / 2.0) * height
            w = box_w * width
            h = box_h * height
            lines.append(
                f"{frame_id},{int(track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1"
            )
            frames_with_tracks.add(frame_id)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "frames": len(image_paths),
        "frames_with_tracks": len(frames_with_tracks),
        "rows": len(lines),
        "image_width": width,
        "image_height": height,
    }


def run_sequence(model: YOLO, sequence: str, max_frames: int | None) -> dict:
    image_dir = DATA_ROOT / sequence / "img1"
    raw_root = EVAL_ROOT / "ultralytics_raw" / sequence
    raw_root.mkdir(parents=True, exist_ok=True)

    # The tracker name is deliberately explicit for auditability.
    results = model.track(
        source=str(image_dir),
        tracker="bytetrack.yaml",
        persist=False,
        stream=True,
        save_txt=True,
        project=str(raw_root),
        name="labels",
        exist_ok=True,
        conf=0.3,
        verbose=False,
        max_det=300,
    )
    for index, _ in enumerate(results, start=1):
        if max_frames is not None and index >= max_frames:
            break

    label_dir = raw_root / "labels" / "labels"
    if not label_dir.exists():
        # Ultralytics versions may place labels one level higher.
        label_dir = raw_root / "labels"
    stats = convert_labels(label_dir, image_dir, EVAL_ROOT / "data" / f"{sequence}.txt", max_frames)
    shutil.copy2(EVAL_ROOT / "data" / f"{sequence}.txt", TRACK_EVAL_ROOT / "data" / f"{sequence}.txt")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke-test limit per sequence")
    parser.add_argument("--sequences", nargs="+", default=["191", "192", "193"])
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {MODEL_PATH}")
    (EVAL_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (TRACK_EVAL_ROOT / "data").mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    sequence_stats = {}
    for sequence in args.sequences:
        print(f"========== ByteTrack sequence {sequence} ==========")
        sequence_stats[sequence] = run_sequence(model, sequence, args.max_frames)
        print(sequence_stats[sequence])

    manifest = {
        "algorithm": "ByteTrack",
        "implementation": "Ultralytics ByteTrack tracker config",
        "tracker_config": "bytetrack.yaml",
        "detector": str(MODEL_PATH),
        "sequences": args.sequences,
        "max_frames": args.max_frames,
        "confidence": 0.3,
        "trackeval_preprocessing": False,
        "sequence_stats": sequence_stats,
    }
    (EVAL_ROOT / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"ByteTrack outputs written to {EVAL_ROOT}")


if __name__ == "__main__":
    main()
