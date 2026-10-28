# 基于深度学习的口腔荧光图像辅助诊断
 
**北京大学口腔医院 | 本科生创新实践项目**
 
## 项目概述

### 当前代码存档说明（2026-10-28）

最新整图二分类实验记录见 `docs/experiments/OR-W-261027-001.md` 和
`OR-W-261027-002.md`。对应源码快照保存在
`experiments/oral_v7_resnet18_baseline_20261026/code/`，本次仅纳入源码、配置和测试，
不包含数据包、训练输出和模型权重。该快照仍引用本地交付包的相对路径，
且数据构建测试所需的 `build_whole_image_classification.py` 尚缺失，不能将其视为完整可复现交付包。
根目录现有 YOLO 和旧版二分类入口继续作为历史实现保留。

新版标签为 `normal_candidate` / `lesion_candidate`；来源组尚未核验为真实患者，
不将来源组独立评价表述为患者独立评价。旧版与新版实验结果不可直接混合比较。
 
针对北京大学口腔医院提供的口腔荧光图像临床数据集，采用深度学习**目标检测**方法
（YOLO 系列）实现口腔黏膜病变的自动检测与分类，并结合可视化技术提供可解释性
诊断依据。最终产出学术论文、软件程序与研究报告。
 
## 任务定位
 
- **任务类型**：目标检测（bounding box）
- **标注格式**：PASCAL VOC XML → 转为 YOLO txt
- **类别**：见 `configs/dataset.yaml`（`l` = 阳性病例，编码1；`n` = 阴性病例，编码0）
- **数据划分**：按**患者**划分，避免同一患者图片跨子集造成数据泄露
 
## 目录结构
 
```
├── configs/            # 配置
│   ├── dataset.yaml    #   数据构建配置（类别映射、划分比例）
│   └── train.yaml      #   训练超参
├── src/                # 核心代码
│   ├── data/           #   voc.py(解析) / build_dataset.py(构建) / analyze.py(探索)
│   ├── train/          #   yolo_train.py
│   ├── eval/           #   yolo_eval.py
│   ├── infer/          #   predict.py
│   └── utils/          #   paths.py / seed.py
├── scripts/            # 命令行入口（薄封装）
│   ├── analyze_data.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data/               # 原始数据（不入库）
│   └── raw/{images,annotations}/
├── datasets/           # 生成的 YOLO 数据集（不入库，硬链接）
├── experiments/        # 每次训练的实验留档
├── reports/            # 指标与图表
├── app/                # 演示系统（后期）
├── docs/               # 文档（文献综述、技术方案等）
└── archive/            # v1 分类脚手架（已归档，保留备查）
```
 
## 快速开始
 
### 1. 安装依赖
 
```bash
pip install -r requirements.txt
```
 
> 需要 GPU 训练：请确认安装的是 CUDA 版 torch（`python -c "import torch; print(torch.cuda.is_available())"` 应为 True）。
 
### 2. 数据探索
 
```bash
python scripts/analyze_data.py
```
 
### 3. 构建 YOLO 数据集
 
```bash
python scripts/prepare_data.py
```
 
生成 `datasets/oral_fluorescence/`（含 `data.yaml`）。
 
### 4. 训练
 
```bash
# 用小模型快速跑通
python scripts/train.py --model yolov8n.pt --epochs 50 --name baseline_n
```
 
### 5. 评估
 
```bash
python scripts/evaluate.py --weights experiments/baseline_n/weights/best.pt --split test
```
 
### 6. 推理
 
```bash
python scripts/predict.py --weights experiments/baseline_n/weights/best.pt --source path/to/image.jpg
```

## 二分类基线

当前新增了按患者切分的 `l/n` 二分类流程。其中 `l` 为阳性病例，
`n` 为阴性病例。

标签编码已统一为 `n = 0`（阴性）、`l = 1`（阳性）。旧版实验曾使用
`l = 0`、`n = 1`，其敏感度、召回率和阈值结果仅作为历史记录保留，
不能直接当作 `l` 阳性结果复用。

```bash
# 构建匿名清单并训练 ResNet18
python scripts/train_binary.py --config configs/binary.yaml

# 在验证集选择阈值后评估测试集
python scripts/evaluate_binary.py --config configs/binary.yaml

# 仅使用有有效框的图片
python scripts/train_binary.py --config configs/binary_annotated.yaml
python scripts/evaluate_binary.py --config configs/binary_annotated.yaml

# 颜色/纹理逻辑回归对照
python scripts/train_binary_features.py --config configs/binary_annotated.yaml

# 使用已有清单复跑并生成训练过程可视化
python scripts/train_binary.py --config configs/binary_annotated_seed43.yaml
python scripts/evaluate_binary.py --config configs/binary_annotated_seed43.yaml
python scripts/visualize_binary.py --config configs/binary_annotated_seed43.yaml

# 冻结主干、仅训练线性分类头
python scripts/train_binary.py --config configs/binary_frozen_seed46.yaml
python scripts/evaluate_binary.py --config configs/binary_frozen_seed46.yaml
python scripts/visualize_binary.py --config configs/binary_frozen_seed46.yaml

# 最小测试
python -m unittest tests.test_binary_pipeline
```

当前测试集患者级 AUROC 在 0.61 至 0.64 之间，尚不具备稳定区分能力。
实验过程、结果和局限见 `docs/experiments/OR-B-261019-001.md`。

项目全过程文字日志见 `docs/项目全过程日志.md`，后续按日期持续追加。
 
## 团队分工
 
| 角色 | 专业 | 主要负责 |
|------|------|----------|
| 队员1 | 计算机科学 | 模型、训练 pipeline、论文 |
| 队员2 | 物理光电 | 荧光图像预处理、可解释性、原型系统 |
 
## 目标成果
 
- 学术论文（SCI）
- 可演示原型系统
- 项目研究报告
