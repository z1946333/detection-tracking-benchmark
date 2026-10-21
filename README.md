# Pig Detection and Multi-Object Tracking Benchmark

猪目标检测与多目标跟踪实验仓库。检测实验使用统一的测试集评估六种检测器；跟踪实验使用同一 YOLOv8n 检测器，在 MOT 序列 191、192、193 上比较六种跟踪器。

## 已完成的检测实验

| Detector | AP@[.50:.95] | AP50 | AP75 |
|---|---:|---:|---:|
| Faster R-CNN | 76.22 | 98.34 | 93.86 |
| RetinaNet | 76.46 | 98.52 | 94.24 |
| SSD300 | 68.44 | 96.61 | 84.77 |
| YOLOv5n | 77.63 | 98.88 | 95.08 |
| YOLOv8n | 79.74 | 98.76 | 95.90 |
| YOLOX | 72.34 | 98.80 | 92.58 |

预测文件和统一评估脚本位于 `evaluation/detection/` 与 `tools/evaluate_coco.py`。

## 已完成的跟踪实验

| Tracker | HOTA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|
| ByteTrack | 24.129 | 23.377 | 68.197 | 11493 |
| BoT-SORT | 72.768 | 91.271 | 90.416 | 5 |
| OC-SORT | 71.974 | 89.399 | 90.920 | 19 |
| Deep OC-SORT (adapted) | 68.620 | 85.347 | 90.869 | 24 |
| StrongSORT (BoxMOT) | 70.704 | 88.169 | 85.668 | 11 |
| HybridSORT (BoxMOT) | 70.953 | 87.772 | 90.862 | 26 |

跟踪结果位于 `evaluation/tracking/`，每个正式结果目录包含 MOT `data/*.txt`、TrackEval 汇总和运行记录。

## 运行环境

- Windows 10/11
- Python 3.9
- PyTorch 2.5.0+cu118
- Ultralytics 8.4.182
- TrackEval（项目内评估脚本）
- BoxMOT 10.0.74/10.0.75 compatibility implementation

原始数据集不随仓库提交。请先按 `docs/data_inventory.md` 准备数据，再修改脚本中的本机路径。

## 主要入口

```text
scripts/prepare_detection_data.py
scripts/train_yolov8.py
scripts/run_bytetrack.py
scripts/run_deep_ocsort.py
scripts/run_boxmot.py
tools/evaluate_coco.py
```

TrackEval 使用当前项目自定义 MOT 配置，类别显示为 `pedestrian`，实际目标类别为 pig，并使用 `DO_PREPROC=False` 兼容自定义标注。Deep OC-SORT、StrongSORT 和 HybridSORT 的适配细节见 `docs/`。
