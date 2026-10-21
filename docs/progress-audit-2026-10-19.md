# 检测与跟踪实验进度审计

检查日期：2026-10-19。此文件记录检查时的状态，训练进度会继续变化。本次没有中断、重新启动训练，也没有改动训练脚本、数据和标签。

## 交付要求

分别完成至少六种检测算法和六种 2021 年以来跟踪算法的比较；检测标注由脚本转换；统一评估并绘制三线表；Word 只写实验设置、对比实验讨论、失败案例和局限。沿用此前 CIFAR-10 项目的组织方式，将代码、配置、真实运行记录、指标、图片、报告和必要权重上传到 GitHub，并核验远端文件及提交。README 面向导师，本机路径与操作步骤放 docs/local-guide.md。

## 已核实的完成项

- 检测原始数据：D:/Datasets/dataset_A/pig-objection detection，1015 个 XML，共 9735 个 pig 目标。
- scripts/prepare_detection_data.py 已实现 XML 到 YOLO 的转换；configs/detection/data.yaml 指向 D:/Datasets/processed_detection。
- 训练集 710 张、6814 个框；验证集 152 张、1470 个框；测试集 153 张、1451 个框。
- 本次基础检查：图片和标签按文件名配对，无缺失或孤立标签；五列数值标签中未发现非正尺寸及超出归一化图像范围的框；三个集合没有同名图片交叉。这不等于已排除近重复图像和同源视频泄漏，也不替代人工叠框检查。
- YOLOv8 有试跑文件及正式训练输出。正式配置：YOLOv8n，100 epochs，imgsz=640，batch=4，workers=4，seed=42，CUDA，AMP；检查时主训练进程 PID 68452 存在，CSV 已记录到 epoch 16。
- experiments/detection/yolov8n/weights 下已有 best.pt 和 last.pt，但训练未完成，不作为最终归档。
- 跟踪原始数据：D:/Datasets/dataset_B/PIG_MOT_TEST_DATA，含序列 191–197 的 MP4、gt/gt.txt、labels.txt。标签为 pig，GT 每行九列，包含 frame 和 ID。
- 七段视频均可读取元数据，1920×1080、30 FPS。视频帧数分别为 1737、1883、1904、1813、1718、1883、1811，与各 GT 最大帧号一致；各序列 ID 数分别为 10、10、8、7、7、7、7。仍需验证逐帧时间对齐、框与图像贴合、重复 ID 和忽略规则。
- 实际训练环境分发版本：torch 2.5.0+cu118，torchvision 0.20.0+cu118，ultralytics 8.4.132，numpy 2.0.2，opencv-python 5.0.0.93。不能把工作目录名称中的 8.3.163 当成实际安装版本。

## 优先解决的数据与可复现问题

1. **划分来源未审计。** 转换脚本直接随机打乱单张 XML，没有按视频、猪舍或采集场景分组。需要追溯来源，确认近重复帧是否跨集合。未确认前，本轮只可作为当前随机图片划分下的结果；若发现泄漏，应保存本轮、另建分组版本并重新训练，不覆盖当前数据。
2. **划分不能只依赖 seed。** glob 结果未排序，也没有固定划分清单；重跑不保证在所有机器上得到相同集合。应先保存现有划分清单，再为正式方案建立版本化清单。当前训练进行中禁止重跑转换。
3. **转换验收不完整。** 需要图像实际尺寸与 XML 一致性、坐标起点规范、difficult/忽略框处理、源框与转换框一致性、多场景叠框审查。当前脚本遇到未知类别或无 pig 图片会跳过、重复执行不清理旧划分；当前数据数量核对通过不代表工具对其他数据安全。
4. **缺 COCO 统一评估标注和各框架适配。** 当前仅有 YOLO 标签；六算法比较需要固定类别映射、统一原图坐标和共同评估配置。
5. **实验溯源尚不完整。** 需要冻结依赖、源码版本、初始化权重哈希、划分及配置；当前训练脚本 exist_ok=True 且 resume=False，直接再运行可能覆盖同目录结果。当前任务结束后应按算法/seed/run_id 归档，再另开新任务目录。
6. **跟踪数据名称含 TEST。** 需确认提供者是否有训练或验证序列。不能直接在这七段正式测试视频上选择最优阈值再报告测试成绩；缺验证集时，应明确约定序列级留出协议及局限，或采用预先固定参数并披露来源。
7. **两个任务间的来源重叠待查。** 检测训练图片是否抽自这些跟踪测试视频尚未确认；如重叠，须调整检测训练数据或测试协议，避免跨任务泄漏。

## 尚未完成的实验

| 工作 | 当前状态 | 下一项验收 |
|---|---|---|
| YOLOv8n 正式训练 | 运行中 | 结束日志、完整 CSV、最终 checkpoint 和验证输出 |
| YOLOv8n 独立测试 | 未找到证据 | 固定方案后对 test 评估，保存 JSON 和预测 |
| YOLOv5、YOLOX、SSD、RetinaNet、Faster R-CNN | 未找到代码适配或结果 | 每种先试跑，再正式训练与共同评估 |
| 检测统一指标与效率测量 | 未完成 | COCO AP50:95/AP50/AP75、参数、统一硬件延迟 |
| 跟踪数据整理和适配 | 仅有原始视频与 GT | 帧与 GT 对齐、评估器类别映射、序列划分 |
| 固定检测器和检测缓存 | 未找到 | 保留低分框，记录权重和检测设置 |
| 六种跟踪算法 | 未找到 | ByteTrack、BoT-SORT、OC-SORT、Deep OC-SORT、StrongSORT、Hybrid-SORT 的轨迹 |
| TrackEval 评估 | 未找到 | HOTA/IDF1/MOTA/IDSW、DetA/AssA、FP/FN、逐序列结果 |
| 多次训练与不确定性 | 尚未实施 | 条件允许时三 seeds；否则明确单次局限 |
| 对比表、失败案例和 Word | 未找到正式结果文件 | 由原始指标生成表格和有证据的讨论 |

跟踪应先固定一个检测器，再对比六种跟踪器；不要一开始运行 36 个组合。ReID 行人权重对猪的跨域适用性需披露。StrongSORT 在线与 StrongSORT++ 离线后处理应分组，不能混报。

## GitHub 发布缺项

检查时目录不是 Git 仓库，git status 和 git remote 均报 not a git repository。因此不能确认已关联任何远端或已上传。本项目根目录还没有 README.md、requirements.txt、.gitignore；docs/data_inventory.md 是空文件，models、tools、tests、evaluation、configs/tracking、third_party 未发现实际文件。

待补：

- 面向导师的 README：真实状态、数据和方法、结果表、复现命令、报告和权重链接、引用与限制。
- 版本化配置、依赖锁定和源码来源清单；机器路径放本机指南。
- .gitignore：排除原始大数据、缓存、环境、凭据；不要一并忽略全部 experiments 和 evaluation。
- 建立并关联独立的新仓库，保存代码与配置阶段提交；训练结束后的稳定结果另外归档提交。
- 训练中正在写入的 best.pt、last.pt 不应当作完整权重快照提交；结束后选择所需权重，超过普通 Git 限制时使用 LFS 或 Release，并核验下载与哈希。
- 最终对照 git rev-parse HEAD、origin/main、git ls-remote 及远端文件树，浏览器核验报告、表格、图、指标、轨迹和权重。

## 推荐顺序

当前让 YOLOv8 继续运行；同时只读核查数据来源和划分、补全溯源与发布文档。训练结束后保存本轮完整成果，依据划分审计决定能否纳入正式主表。然后完成其余五种检测算法、固定检测缓存和六种跟踪评估，最后生成对比表、Word 和远端可验证的最终仓库。

本次仅执行进度审计，没有初始化或推送 Git，也没有提前运行测试集、启动其他训练、修改当前训练配置。

## 2026-10-21 后续运行记录

- Hybrid-SORT 第三方核心的 `predict()` 分数读取错误已修复；回归测试 `tests/test_hybrid_sort_adapter.py` 通过。
- `scripts/run_hybrid_sort.py` 已完成序列 191、192、193，完整帧数为 1737、1883、1904，输出分别为 17387、18490、13353 行。
- 采用 TrackEval MotChallenge2DBox、`DO_PREPROC=False`、HOTA/CLEAR/Identity，评估结果保存在 `evaluation/tracking/hybridsort_full_20261021/`。合计：HOTA 68.907、IDF1 84.526、MOTA 90.782、IDSW 37。
- 本次 TrackEval 使用 `pedestrian` 作为内部类别名，仅为兼容评估器；数据真实类别是 pig。`DO_PREPROC=False` 是当前临时兼容设置，正式报告前应完成 pig 自定义 adapter 或在方法中明确这一限制，不能把该结果称为官方 MOT17 结果。
- 初步与已有 Deep-OC-SORT 适配版汇总相比，Hybrid-SORT 的 HOTA 高 0.287 个百分点、IDF1 低 0.821 个百分点、MOTA 低 0.087 个百分点；这是同一当前协议下的观察，不是显著性结论。
