# PyCharm 缓存训练配置

缓存目录：`data/cache/oral_binary_640_v1/`。保留原冻结交付包，缓存只生成 train 893 张、val 112 张，不生成测试缓存。图片数组按 split/模态分目录，清单保存标签、来源哈希、数组哈希、尺寸、算法版本和实现哈希。

训练顺序改为先等比缩小（最长边640），再在线随机翻转/旋转（扩展画布），最后等比缩放补边、归一化；不裁剪。此版本与原图先旋转的流程不完全等价，需作为独立实验。验证集缓存原流程生成的 float32 输入，已逐张验证完全相等。

## 在 PyCharm 中选择

- `Cache - Verify and Benchmark (no training)`：只验证缓存和测量加载速度，报告写入 `reports/cache/640_v1_pycharm_verification.json`。
- `Ablation - Cache v1 ResNet50 (640, workers 0)`：独立ResNet50/none/CE实验，640、batch4、最多100轮、patience20，其余优化器及随机种子沿用原配置。运行记录带 `_cache_v1` 后缀，生成新的报告目录；不接续旧检查点。

对应配置文件为 `.run/cache_verify.run.xml`、`.run/ablation_cache_v1.run.xml`。如果列表未显示，打开 Run → Edit Configurations 查看项目配置。

2026-10-29 用户授权切换后，已从 PyCharm 停止旧流程并启动缓存优化 ResNet50 独立实验。旧实验保存至58轮，保留last.pt、best.pt和优化器状态；新实验从预训练权重重新开始。当前批次reports/ablation/20261029_114509，看板http://127.0.0.1:13949/已切换。不要再点击运行而启动第二份训练。

## 重建规则

不覆盖既有缓存目录。源清单、尺寸、实现哈希或缓存负载不匹配时直接报错，使用新的版本目录重建并同步 PyCharm 参数。检查点记录预处理版本和缓存清单哈希，拒绝跨版本原地续训。

首次构建使用项目 `.venv/Scripts/python.exe` 运行 `scripts/prepare_image_cache.py`，参数包括 `--data` 原清单路径、`--cache-dir` 新目录、`--build`、`--report` 新报告路径。已存在的缓存只验证/测速时不要加 `--build`。

缓存用 numpy 文件按需映射读取，不强制全量驻留内存；仍存在文件读取，但省去反复JPG解码和大原图旋转。首次启动会校验缓存哈希，因此预检耗时不能当作每轮加载耗时。原数据完整性预检仍会读取冻结包（含测试文件的结构/哈希检查），不运行测试推理或计算测试性能。

## 验证记录

原始测速报告：`reports/cache/640_v1_verification.json`；带资源采样的复测：`reports/cache/640_v1_resources.json`；训练入口只预检：`reports/cache/cache_preflight.json`。测速是在原正式训练并行运行时进行的CPU数据加载测试，包含Windows工作进程启动/退出开销，不代表GPU模型训练提速或分类效果。

完整工作记录见 `../experiments/oral_v7_resnet18_baseline_20261026/code/docs/PROJECT_WORK_LOG.md`。


## 八组正式实验队列

在PyCharm选择 `Ablation - Cache Formal Eight (reuse completed baseline)`。配置位于 `.run/ablation_cache_formal_eight.run.xml`，核对既有缓存ResNet50基线的配置、数据及缓存哈希后计入第0组；顺序运行第1至7组，不重复基线。当前批次为 `reports/ablation/20261029_141726`，看板仍为 http://127.0.0.1:13949/ 。后续模型依次为EfficientNet-B3、DenseNet121、ResNet50+SE、ResNet50+CBAM及ResNet50的weighted CE、label smoothing、focal；640、batch4、workers0、seed42、最多100轮、patience20。当前队列已启动，请勿重复点击Run。
