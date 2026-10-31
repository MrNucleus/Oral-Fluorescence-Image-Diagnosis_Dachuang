# PyCharm 架构、注意力与损失消融

在项目右上角运行配置下拉框选择以下入口。如果尚未显示，使用 File → Reload All from Disk，或重新打开项目。配置保存在根目录 `.run/`；解释器使用 `.venv/Scripts/python.exe`。

| 配置 | 行为 |
|---|---|
| Ablation - Live Results Monitor | 每3秒读取已完成的验证结果；不启动训练。适合查看当前已启动的试验 |
| Ablation - Pilot All (320, 1 epoch) | 顺序执行8组，每组1轮，320完整视野；仅作软件与初始收敛检查 |
| Ablation 0 ... 7 | 运行指定单组，640完整视野、batch 4、最多100轮、patience 20 |

如果后台已有整套试验，不要同时点击另一个训练入口争用显存。可以运行监视器，或等当前试验结束后再开启正式实验。

## 实验表

| 索引 | 编码器 | 新增注意力 | 损失 |
|---|---|---|---|
| 0 | ResNet50 | 无 | CE |
| 1 | EfficientNet-B3 | 无额外模块 | CE |
| 2 | DenseNet121 | 无 | CE |
| 3 | ResNet50 | SE | CE |
| 4 | ResNet50 | CBAM | CE |
| 5 | ResNet50 | 无 | 加权CE |
| 6 | ResNet50 | 无 | 标签平滑CE，0.1 |
| 7 | ResNet50 | 无 | Focal，gamma=2 |

正式消融以 ResNet50 为统一基线，架构对照为 EfficientNet-B3 和 DenseNet121。所有组使用相同冻结train/val、seed 42、几何增强、AdamW和学习率0.001。EfficientNet原架构含自身的SE，表中的“无额外模块”不是删除内部SE。注意力插入末端特征图与池化之间，不使用XML框或ROI裁剪。

加权CE权重仅按所选训练集统计：N/(2*n_class)。不同训练损失之间始终使用统一、无权重、无平滑的验证交叉熵选模；AUROC等指标同时保留。测试集不会被用于这套对比的模型选择或性能计算。数据预检会检查整个冻结包的完整性，区别于测试集性能评价。

本轮所有组统一 batch 4；历史 ResNet50 基线使用 batch 8，因此编号0需重新运行作为本轮对照，不能直接用历史结果替代。

本轮是分因素实验，不是全部架构×注意力×损失的全组合。先用正式验证结果筛选，再为候选补充多种子和组级验证。1轮320试验不能用于宣称优于历史640基线。

## 查看代码与结果

- 总入口：`scripts/run_architecture_ablation.py`
- 实时监视器：`scripts/watch_ablation.py`
- 模块：`experiments/oral_v7_resnet18_baseline_20261026/code/src/oral_fluorescence_detection/ablation.py`
- 模型/损失 CLI：嵌套源码中的 `scripts/train_image_classifier.py`，可传 `--model`、`--attention`、`--loss`、`--seed`。
- 支持额外编码器 EfficientNet-B0；可通过 CLI 独立运行。
- 每次验证汇总：`reports/ablation/<时间>/validation_summary.json`
- 完整日志和检查点：嵌套源码的 `outputs/image_classifier_runs/ablation_*`。
- 初次 DenseNet121 使用 torchvision 官方权重，未缓存时会自动下载。

模型权重保存新增注意力和损失元数据；评估可重建相同结构，旧的无注意力检查点继续兼容。新运行生成 training_config.yaml，续训脚本使用它保持对应配置。

验证命令：`.venv/Scripts/python.exe -m unittest discover -s tests -v`。

## 试跑完成后启动正式训练（2026-10-28）

在 PyCharm 右上角选择 `Ablation - Formal All (640, 100 epochs)` 并点击绿色运行按钮。配置文件位于 `.run/ablation_formal_all.run.xml`，顺序运行全部8组。

参数仅保留 `--data` 及原数据路径，不带 `--pilot`；正式设置为640完整视野、batch 4、seed 42、AdamW学习率0.001、最多100轮、patience 20。从ImageNet预训练权重开始，不接续320试跑权重；结果使用新的时间戳目录。测试集不计算性能。

若当前配置列表未出现新名称，可在现有试跑配置的 Edit Configurations 中只删除 Parameters 开头的 `--pilot`，保留 `--data` 和完整数据路径，然后 Apply → Run。不要同时启动两份训练。

当前本地看板读取的是已结束的320试跑，不能把它当作正式训练进度。正式训练先看 PyCharm 底部 Run 窗口的 `phase/train`、`batch`、`epoch` 和 `val_metrics` 输出。
