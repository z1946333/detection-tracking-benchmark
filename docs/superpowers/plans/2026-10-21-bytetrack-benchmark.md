# ByteTrack Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a traceable ByteTrack run using the same YOLOv8 detector, sequences 191–193, and TrackEval protocol used by the other trackers.

**Architecture:** `scripts/run_bytetrack.py` calls Ultralytics `bytetrack.yaml`, converts each saved normalized tracking label to MOTChallenge format, and writes a run manifest beside the outputs. TrackEval then evaluates the three MOT files and stores HOTA, CLEAR, and Identity summaries under `evaluation/tracking/bytetrack_20261021`.

**Tech Stack:** Python 3.9, Ultralytics YOLO, OpenCV, TrackEval, PowerShell.

**Spec:** `docs/progress-audit-2026-10-19.md` and the project tracking comparison requirement.

## Global Constraints

- Use `D:\conda_envs\yahui_torch\python.exe`.
- Use `experiments/detection/yolov8n/weights/best.pt` for the shared detector.
- Evaluate only sequences 191, 192, and 193, matching the existing tracker results.
- Keep raw data outside the repository and write MOT files with ten columns.
- Do not overwrite any existing tracker result directory.

## Review Focus

- The script must explicitly use `bytetrack.yaml`, not `botsort.yaml`.
- Each sequence must produce a separate `191.txt`, `192.txt`, and `193.txt`.
- Track IDs must be preserved and MOT coordinates must be pixel coordinates.
- The output directory and manifest must identify ByteTrack unambiguously.

### Task 1: Add the ByteTrack runner

**Files:**
- Create: `scripts/run_bytetrack.py`
- Create: `docs/superpowers/plans/2026-10-21-bytetrack-benchmark.md`

- [x] Create the runner with explicit detector, sequences, tracker config, conversion, and manifest paths.
- [x] Compile the runner with `py_compile`.
- [x] Run a one-frame smoke check before the full experiment.

### Task 2: Run and evaluate

- [x] Run `scripts/run_bytetrack.py` with the project Python interpreter.
- [x] Verify all three MOT files and ten-column rows.
- [x] Run TrackEval with `HOTA CLEAR Identity` and `DO_PREPROC=False`.
- [x] Verify `pedestrian_summary.txt` and record the metrics in `docs/bytetrack-benchmark.md`.
