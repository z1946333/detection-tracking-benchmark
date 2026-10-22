# 检测模型效率测量

脚本：`scripts/benchmark_detection_efficiency.py`

测量协议：

- RTX 3050 Laptop GPU，batch=1
- 全部 153 张测试图像
- 每个模型预热 10 次，正式计时 153 次
- CUDA 同步后计时模型调用，不包含磁盘读写
- YOLOv5n、YOLOX、YOLOv8n 使用 640 px 输入
- Faster R-CNN、RetinaNet、SSD300 使用现有 torchvision 输入处理
- 峰值显存包含模型加载和推理阶段

结果文件：

- `evaluation/detection/efficiency_benchmark_20261022.csv`
- `evaluation/detection/efficiency_benchmark_20261022.json`

本次结果用于报告表 2。不同模型的输入预处理边界不同，因此延迟适合作为本项目推理管线下的工程参考；若需要严格部署比较，应进一步统一输入尺寸、预处理和后处理边界。

