# Deep OC-SORT 调用修复与验证

日期：2026-10-21。

## 根因

1. 原 scripts/run_deep_ocsort.py 实际导入 trackers.deepsort_tracker.deepsort.DeepSort，这是 DeepSORT，不能标成 Deep-OC-SORT。作者 main.py 使用的是 trackers/integrated_ocsort_embedding/ocsort.py 的 OCSort。
2. 本次实际复现报错为 No module named torchreid。相对导入本身并不要求 pip 安装整个包；必须在保留包上下文时导入。原脚本不断添加 sys.path 并捕获 ImportError，会掩盖真正依赖错误，且多个项目都有 trackers 目录，存在串包风险。
3. 用户本地已修改的 reid_model.py 构建 osnet_x0_25(num_classes=1000)，已有权重 classifier.weight 为 (1041,512)，classifier.bias 为 (1041,)。strict=False 也不能忽略这些同名张量尺寸不匹配。

## 实际修复

- 原脚本已备份到 docs/run_deep_ocsort.before-fix-2026-10-21.txt；上游目录及其他三个算法的输出未改动。
- 在 src/vendor/deep_ocsort 保存本地作者 integrated 实现的核心源码与 OSNet 定义，并保留许可证。不再依靠多个同名 trackers 包，不需要安装整个 torchreid 训练工具链。
- 跟踪核心仅改两处：注入真实 ReID 提取器；CMC 改为逐帧 sparse optical flow，避免硬编码读取 MOT17/DanceTrack 的预计算相机运动文件。关联逻辑保持该本地作者实现。
- OSNet 根据权重头的 1041 类构建并 strict=True 加载全部权重，eval 模式输出 512 维特征。1041 是预训练分类头尺寸，不是当前猪类别数。使用 RGB、256×128 裁剪、ImageNet 归一化及 L2 特征归一化。
- 使用 Deep OC-SORT 的自适应关联和外观更新；embedding_off=False、aw_off=False、grid_off=True、new_kf_off=True、w_association_emb=0.75、aw_param=0.5。没有启用离线插值。
- YOLO 原图 xyxy 坐标传入跟踪器时保持 scale=1；空检测帧继续调用 update。
- 新结果写 evaluation/tracking，而非被 .gitignore 排除的 third_party 下；已有输出目录非空时拒绝覆盖。run.json 保存设置、源码及权重 SHA256、完成状态和 ReID 调用次数。

## 必须准确命名

本配置称为 **Deep OC-SORT + OSNet-x0.25（适配版）**。保留真正 Deep OC-SORT 关联机制，但外观模型不是论文的 OSNet-AIN/SBS 原配置；CMC 为自定义视频在线估计。现有文件名标记 MSMT17，权重哈希已经记录，但文件的官方来源还需补充。不能将本配置成绩声称为论文原配置复现，也不能用它断言论文方法在猪数据上的最佳表现。

若导师要求严格原配置，需要取得作者指定的 ReID 权重并按其对应网络和预处理重新配置。行人外观模型跨域用于猪的限制须在报告中说明。

## 已完成的验证

- 三项回归测试通过：真实特征有限、单位范数且依赖图像内容；连续帧保持身份并在空检测帧推进年龄；不匹配权重拒绝加载。
- 序列 191 前 100 帧实测完成，100 次真实 ReID 调用，输出 1009 条轨迹记录，无同帧重复 ID，宽高均为正。
- 记录：evaluation/tracking/deep_ocsort_smoke_20261021/run.json。
- 此为试跑，不是完整序列成绩。本次没有生成正式 TrackEval 分数，也没有重跑其他方法。

## 在 VS Code 运行完整三个序列

终端类型 PowerShell：

```powershell
Set-Location 'D:\GitHub\detection-tracking-benchmark'
& 'D:\conda_envs\yahui_torch\python.exe' scripts/run_deep_ocsort.py --sequences 191 192 193
```

等到最后显示 Completed，检查 evaluation/tracking/deep_ocsort_osnet_x025/run.json：status=completed、smoke_test=false，三个序列帧数应为 1737、1883、1904。已有非空输出时使用新的 --output 目录；不要删除旧结果来强行重跑。

然后运行现有 TrackEval，保持与前面算法相同的 GT 和评估设置。针对默认输出的命令如下（单行复制）：

```powershell
& 'D:\conda_envs\yahui_torch\python.exe' third_party/TrackEval/scripts/run_mot_challenge.py --GT_FOLDER third_party/TrackEval/data/gt/mot_challenge/PIG-train --TRACKERS_FOLDER evaluation/tracking --TRACKERS_TO_EVAL deep_ocsort_osnet_x025 --SKIP_SPLIT_FOL True --BENCHMARK PIG --SPLIT_TO_EVAL train --SEQ_INFO 191 192 193 --METRICS HOTA CLEAR Identity --DO_PREPROC True --USE_PARALLEL False
```

这条完整评估命令尚未运行，需等待完整跟踪输出。不要把 100 帧试跑送入 1737 帧 GT 后把缺失帧造成的分数当正式结果。当前 TrackEval 用 pedestrian 作为 class=1 的内部名称；在项目报告中真实类别仍为 pig，不能描述为行人数据。

## 另发现的评估元数据问题

实际抽帧图片为 1920×1080，但现有 191/seqinfo.ini 写成 1280×720。该字段不等于已证明框坐标缩放错误，也未据此否定已有成绩；需要修正元数据并抽查 GT 与原图对齐。本次未改已有评估设置或成绩。前三种算法的“公平比较”仍应另行核验检测阈值、序列、预处理和结果口径，不能仅凭有 TrackEval 数字称为完整复现。

## GitHub 归档

应保留 scripts/run_deep_ocsort.py、src/deep_ocsort_adapter.py、src/vendor/deep_ocsort（含两个许可证及来源哈希）、tests/test_deep_ocsort_adapter.py、此说明及 evaluation/tracking 下真实结果。权重被当前 .gitignore 排除，上传时需提供合法下载来源或 Release/LFS 与 SHA256。本次未执行远端上传。

上游参考：https://github.com/GerardMaggiolino/Deep-OC-SORT

OSNet 参考：https://github.com/KaiyangZhou/deep-person-reid

本地源码来自无 .git 的目录，无法核验上游 commit，不能编造版本号；本地原文件与适配文件 SHA256 见 src/vendor/deep_ocsort/provenance.json。
