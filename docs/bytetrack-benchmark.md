# ByteTrack 实验记录

## 运行入口

```powershell
cd D:\GitHub\detection-tracking-benchmark
& 'D:\conda_envs\yahui_torch\python.exe' scripts\run_bytetrack.py
```

脚本明确调用 Ultralytics 的 `bytetrack.yaml`，使用统一的 YOLOv8n 检测器权重，并处理 MOT 序列 191、192、193。原始标签、MOT 转换结果和 `run.json` 保存在：

```text
evaluation/tracking/bytetrack_20261021/
```

## TrackEval 命令

```powershell
& 'D:\conda_envs\yahui_torch\python.exe' third_party\TrackEval\scripts\run_mot_challenge.py `
  --GT_FOLDER third_party\TrackEval\data\gt\mot_challenge\PIG-train `
  --TRACKERS_FOLDER third_party\TrackEval\data\trackers\mot_challenge\PIG-train `
  --TRACKERS_TO_EVAL my_bytetrack `
  --SKIP_SPLIT_FOL True --BENCHMARK PIG --SPLIT_TO_EVAL train `
  --SEQ_INFO 191 192 193 --METRICS HOTA CLEAR Identity `
  --DO_PREPROC False --USE_PARALLEL False --PLOT_CURVES False
```

## 当前真实结果

| HOTA | IDF1 | MOTA | IDSW |
|---:|---:|---:|---:|
| 24.129 | 23.377 | 68.197 | 11493 |

这是当前检测阈值和 Ultralytics ByteTrack 配置在本项目猪数据上的结果，不是 MOT17/MOT20 官方成绩。TrackEval 当前类别显示为 `pedestrian`，实际目标类别是 pig；评估使用 `DO_PREPROC=False` 以兼容项目自定义 MOT 标注。
