# StrongSORT 和 Hybrid-SORT 操作步骤

本文针对本项目的猪多目标跟踪数据。两个算法都采用 tracking-by-detection：每一帧先用同一个 YOLOv8 检测器得到框，再由跟踪器关联身份。为了和已经完成的 ByteTrack、BoT-SORT、OC-SORT 公平比较，六种跟踪器必须使用同一检测器、同一序列、同一帧号和同一评估器。

## 一、先看当前仓库状态

当前 `scripts/run_strongsort.py` 不能直接作为正式入口。它尝试导入 `deep_sort.deepsort.DeepSort`，但 `third_party/StrongSORT/deep_sort` 中只有 `tracker.py`、`detection.py` 等文件，没有这个类；因此报 `No module named deep_sort.deep_sort` 是代码入口不匹配，不是你的数据问题。

Hybrid-SORT 的官方实现也不是一个接收 Ultralytics `YOLO()` 对象的简单类。它的 `tools/run_hybrid_sort_dance.py` 先用 YOLOX 模型检测，再把输出送给 `Hybrid_Sort.update()`。本项目中已经有自己的 YOLOv8 检测器，因此需要写一个适配层：只调用 Hybrid-SORT 的关联器，检测仍由项目统一的 YOLOv8 完成。

## 二、StrongSORT 的组成

StrongSORT 是 DeepSORT 的增强版本，主要包括：Kalman 运动预测、IoU/外观特征门控、匹配级联、ReID 特征更新。StrongSORT++ 另外加入 AFLink 和 GSI 后处理。正式对比表若写“StrongSORT”，先只运行在线 StrongSORT；AFLink、GSI 应另列为 `StrongSORT++`，不能混在同一名称下。

官方源码入口是 `third_party/StrongSORT/strong_sort.py`，它调用 `deep_sort_app.run()`。这个入口要求每个序列先有一个 `.npy` 检测文件，矩阵格式为：

```text
frame, detection_id, x, y, width, height, confidence, -1, -1, -1, feature_1, ..., feature_512
```

源码中 `deep_sort_app.py` 的 `create_detections()` 读取第 3–6 列作为框，第 7 列作为置信度，第 11 列以后作为外观特征。因此只生成 YOLO 框而没有外观特征，不能直接喂给 StrongSORT。

## 三、StrongSORT 具体操作

### 1. 新建独立环境

StrongSORT 原作者代码测试环境较旧，不要为了它升级你正在使用的 YOLOv8 环境。先在 Anaconda Prompt 执行：

```powershell
conda create -n pig_strongsort python=3.9 -y
conda activate pig_strongsort
pip install numpy scipy scikit-learn opencv-python filterpy
```

然后确认基础依赖：

```powershell
python -c "import numpy,scipy,cv2,filterpy; print('dependencies ok')"
```

如果安装旧版 `scikit-learn==0.19.2` 失败，不要在 Windows 上反复编译；使用当前可安装版本，并把版本写入实验记录。最终必须记录 `pip freeze > experiments/tracking/strongsort/pip-freeze.txt`。

### 2. 准备序列目录

每个序列应包含：

```text
191/
  img1/000001.jpg ...
  gt/gt.txt
  seqinfo.ini
```

运行 StrongSORT 前先检查图片名从 000001 连续到最后一帧。当前猪视频实际尺寸是 1920×1080，而已有部分 `seqinfo.ini` 曾记录成 1280×720；先修正元数据并抽查框坐标，再进行正式分数计算。不要只改文件名而不检查标注是否也按比例缩放。

### 3. 生成检测和 ReID 特征

每帧执行以下顺序：

1. `YOLO(best.pt)` 推理；只保留 `class_id=0` 的 pig。
2. 将 `xyxy` 转为 MOT 的 `x,y,w,h`。
3. 对每个框从原图裁剪猪区域，送入与权重结构一致的 OSNet/StrongSORT ReID 模型。
4. 得到 512 维特征并做 L2 归一化。
5. 将 `[frame, -1, x, y, w, h, score, -1, -1, -1, feature...]` 写入一个序列的 `.npy`。

注意：第 2 个字段不是跟踪 ID，而是检测占位 ID；StrongSORT 会自行分配轨迹 ID。检测置信度阈值要和另外五种方法相同或在验证集预先固定，低分框是否保留必须记录。

### 4. 调用 StrongSORT 在线主版本

源码的命令行参数会在导入 `opts.py` 时解析，因此必须从 `third_party/StrongSORT` 目录运行：

```powershell
conda activate pig_strongsort
Set-Location 'D:\GitHub\detection-tracking-benchmark\third_party\StrongSORT'
python strong_sort.py MOT17 val --root_dataset 'D:\Datasets\dataset_B\PIG_MOT_TEST_DATA' --dir_save 'D:\GitHub\detection-tracking-benchmark\evaluation\tracking\strongsort' --dir_dets 'D:\GitHub\detection-tracking-benchmark\evaluation\tracking\strongsort_dets'
```

原版 `opts.py` 的 `data` 字典只认识 MOT17/MOT20 的固定序列名称，所以本项目的 191、192、193 不能直接套用这条命令。正式使用前要在项目适配脚本中把 `sequences` 改成 `['191','192','193']`，并让 `dir_dataset/dir_dets/dir_save` 指向本项目路径。不要直接修改第三方源码后忘记记录差异；适配代码放 `scripts/`，改动说明放 `docs/`。

StrongSORT 基础版建议参数：`min_confidence=0.3`、`max_cosine_distance=0.2`、`nn_budget=100`、`max_age=30`、`n_init=3`。这些只是起始值，最终阈值只能用验证序列选择。`--BoT` 是另一种外观特征配置，不能和普通 OSNet 结果混名。

### 5. 检查 StrongSORT 输出

输出每行必须是 10 列：

```text
frame,id,x,y,w,h,score,-1,-1,-1
```

检查命令：

```powershell
& 'D:\conda_envs\yahui_torch\python.exe' tools/check_mot_result.py --input 'D:\GitHub\detection-tracking-benchmark\evaluation\tracking\strongsort\191.txt' --frames 1737
```

如果 `tools/check_mot_result.py` 尚未创建，就在正式运行前补一个检查器，至少验证列数、帧号范围、正宽高、有限数值和同一帧同一 ID 不重复。检查通过后，再复制到 TrackEval 的 tracker 目录。

## 四、Hybrid-SORT 的组成

Hybrid-SORT 是基于 OC-SORT 的在线、无需训练的关联器。它在强线索之外加入弱线索：四个框角的运动方向、置信度状态和目标高度状态，并可用 Byte 低分框二次关联。源码中有两种入口：

- `Hybrid_Sort`：不使用 ReID，适合本项目的主公平比较。
- `Hybrid_Sort_ReID`：额外依赖 FastReID，属于另一个带外观特征的变体，应单独命名为 `Hybrid-SORT-ReID`。

第一次正式实验建议先做 `Hybrid_Sort`，这样与 OC-SORT 的运动关联比较直接，也不会把行人 ReID 权重跨域用于猪而不说明。

## 五、Hybrid-SORT 具体操作

### 1. 安装与导入

在当前 PyTorch 环境先检查基础依赖：

```powershell
Set-Location 'D:\GitHub\detection-tracking-benchmark'
& 'D:\conda_envs\yahui_torch\python.exe' -c "import torch,cv2,scipy,filterpy; print(torch.__version__); print('hybrid dependencies ok')"
```

不要在 Windows 上立即运行 Hybrid-SORT README 中针对 MOT17 的训练命令；那会下载并加载 YOLOX 行人模型，与当前 pig 数据不是同一个检测实验。

适配层需要把 `third_party/HybridSORT` 放入 `sys.path`，导入：

```python
from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort
```

构造一个参数对象，至少提供源码读取的字段：

```python
from types import SimpleNamespace
args = SimpleNamespace(
    track_thresh=0.30,
    TCM_first_step=False,
    TCM_byte_step=False,
    TCM_first_step_weight=1.0,
    TCM_byte_step_weight=1.0,
    low_thresh=0.10,
    use_byte=False,
)
```

不同仓库版本可能还读取 `match_thresh` 或 `hp` 等字段。第一次导入后如果出现 `AttributeError: args.xxx`，查看报错行，把该字段显式加入适配器，并把版本差异记录下来；不要用 `MagicMock` 把错误吞掉。

### 2. 每帧调用

YOLOv8 输出应转成 `N×5` 的 numpy 数组：`[x1,y1,x2,y2,confidence]`。由于 YOLOv8 已经返回原图坐标，给 Hybrid-SORT 的 `img_size` 也要使用原图尺寸，避免二次缩放：

```python
img_h, img_w = image.shape[:2]
dets = np.asarray([[x1, y1, x2, y2, score], ...], dtype=np.float32)
tracks = tracker.update(dets, [img_h, img_w], [img_h, img_w])
```

无检测时仍传入 `np.empty((0,5), dtype=np.float32)`，不能跳过这一帧。返回数组通常为 `[x1,y1,x2,y2,track_id]`，转换为 MOT 输出：

```python
for x1,y1,x2,y2,track_id in tracks:
    f.write(f'{frame_id},{int(track_id)},{x1:.2f},{y1:.2f},'
            f'{x2-x1:.2f},{y2-y1:.2f},1,-1,-1,-1\n')
```

### 3. 参数起点

建议首先固定：`det_thresh=0.30`、`max_age=30`、`min_hits=3`、`iou_threshold=0.30`、`delta_t=3`、`inertia=0.20`、`asso_func='iou'`、`use_byte=False`。在验证序列上只调 `det_thresh`、`iou_threshold` 和 `max_age`，每次只改一个因素并保存配置。

如果要测试论文中的 Byte 弱线索，另建实验名 `hybridsort_byte`，设置 `use_byte=True`，同时保留 `low_thresh=0.1`；不能把它和基础 Hybrid-SORT 共用一个结果目录。

### 4. 运行顺序

推荐先用 100 帧试跑：

```powershell
Set-Location 'D:\GitHub\detection-tracking-benchmark'
& 'D:\conda_envs\yahui_torch\python.exe' scripts/run_hybrid_sort.py --sequences 191 --max-frames 100 --output 'evaluation/tracking/hybridsort_smoke_20261021'
```

试跑通过后，完整运行：

```powershell
& 'D:\conda_envs\yahui_torch\python.exe' scripts/run_hybrid_sort.py --sequences 191 192 193 --output 'evaluation/tracking/hybridsort'
```

本项目的 `scripts/run_hybrid_sort.py` 已经可以调用 YOLOv8 和 Hybrid-SORT 基础关联器。2026-10-21 曾发现第三方 `hybrid_sort.py` 的 `update()` 被错误改写：它丢弃了 `predict()` 的两个分数，又读取不存在的 `self.score` 和未定义的 `simple_score`。现在已恢复为 `pos, kalman_score, simple_score = tracker.predict()`，并在 `tests/test_hybrid_sort_adapter.py` 中覆盖第二帧回归。

先运行 100 帧试跑：

```powershell
Set-Location 'D:\GitHub\detection-tracking-benchmark'
& 'D:\conda_envs\yahui_torch\python.exe' scripts/run_hybrid_sort.py --sequences 191 --max-frames 100 --output 'evaluation/tracking/hybridsort_smoke_fix_20261021'
```

已验证该试跑生成 1009 行、10 列 MOT 结果，没有同一帧重复 ID。它只是适配器试跑，不能用于正式 TrackEval 分数。试跑通过后，再使用新的空目录运行完整 191、192、193 序列；结果目录非空时脚本会覆盖文件，因此正式脚本还应补充“拒绝覆盖”保护。

## 六、TrackEval 评估

先确认 GT 和结果目录结构：

```text
third_party/TrackEval/data/gt/mot_challenge/PIG-train/191/gt/gt.txt
evaluation/tracking/strongsort/data/191.txt
evaluation/tracking/hybridsort/data/191.txt
```

当前 TrackEval 的 MOTChallenge 数据集默认类别叫 `pedestrian`，而本项目真实类别是 `pig`。需要使用项目自定义的 MOT 数据集配置或 adapter，把类别映射、序列列表和 `seqinfo.ini` 明确写入；不能把猪结果称为行人结果。正式命令应与 ByteTrack、BoT-SORT、OC-SORT 完全一致，指标选择 `HOTA CLEAR Identity`。

评估前检查：

1. 191、192、193 的完整帧数分别为 1737、1883、1904。
2. 预测 TXT 没有试跑的 100 帧文件冒充完整结果。
3. 结果坐标是原图 1920×1080 像素，宽高为正。
4. GT 的 frame、ID、类别和可见性规则已经写入报告。
5. 阈值只在验证序列确定；最终测试序列不能反复调参。

最后保存：`pedestrian_summary.txt`、详细 CSV、评估配置、原始 TXT、运行日志和环境版本。比较表中的 StrongSORT 与 Hybrid-SORT 数值只能来自这次猪数据的 TrackEval 输出，不能抄 README 中 MOT17、MOT20 或 DanceTrack 的数字。

## 七、报告中怎么写

建议把方法名写成：

```text
StrongSORT (online, OSNet-x0.25, adapted)
Hybrid-SORT (motion/weak-cue variant, adapted)
```

如果使用 ReID 版本，另列：`StrongSORT-ReID` 或 `Hybrid-SORT-ReID`，并写明权重来源、输入裁剪、跨域问题和额外耗时。不要把 StrongSORT++ 的 AFLink/GSI 后处理放进 StrongSORT 行。

报告讨论时分别说明：

- HOTA/IDF1 上升是否伴随 IDSW 减少；
- MOTA 变化来自 FP、FN 还是关联错误；
- StrongSORT 的外观特征对猪这种跨域目标是否稳定；
- Hybrid-SORT 的弱运动线索在遮挡、交叉和密集目标场景是否减少身份切换；
- ReID、检测、关联和端到端 FPS 是否分别计时。

所有结果完成后，再把脚本、配置、日志、评估结果、表格和 Word 报告一起提交到 GitHub。第三方源码保留许可证，原始视频和网盘数据不要上传。
