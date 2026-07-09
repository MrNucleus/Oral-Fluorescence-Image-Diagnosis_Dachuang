# 基于深度学习的口腔荧光图像辅助诊断

**北京大学口腔医院 | 本科生创新实践项目**

## 项目概述

本项目针对北京大学口腔医院提供的口腔荧光图像临床数据集，研究口腔黏膜病变的AI辅助诊断方法。主要研究内容包括：

1. 构建标准化的口腔荧光图像数据集
2. 研发基于深度学习的分类模型（5类：Normal / OLK / OLP / OSCC / Other）
3. 采用Grad-CAM实现可解释性诊断
4. 开发可演示的原型诊断系统

## 项目结构

```
├── data/               # 数据集
│   ├── raw/            # 原始图像（不修改）
│   ├── processed/      # 预处理后图像
│   └── splits/         # 数据集划分文件
├── src/                # 核心代码
│   ├── config.py       # 全局配置
│   ├── data/           # 数据加载与预处理
│   ├── models/         # 模型定义（ResNet/EfficientNet + CBAM）
│   ├── training/       # 训练循环与损失函数
│   ├── evaluation/     # 评估指标与可视化
│   ├── interpret/      # Grad-CAM可解释性
│   └── utils/          # 工具函数
├── scripts/            # 训练/评估/预处理入口
├── notebooks/          # Jupyter Notebooks
├── docs/               # 项目文档
├── experiments/        # 实验记录
├── reports/            # 评估报告与图表
├── checkpoints/        # 模型权重（gitignored）
└── app/                # 原型系统（后期）
```

## 快速开始

### 环境配置
```bash
pip install -r requirements.txt
```

### 数据准备
1. 将原始图像放入 `data/raw/`，按类别组织子目录
2. 运行数据划分脚本生成 train/val/test 划分
3. （可选）运行预处理脚本进行光照校正
```bash
python scripts/preprocess.py --input-dir data/raw --output-dir data/processed
```

### 训练模型
```bash
python scripts/train.py --backbone resnet50 --exp-name baseline_v1
```

### 评估模型
```bash
python scripts/evaluate.py --checkpoint checkpoints/baseline_v1/best_model.pth
```

## 团队分工

| 角色 | 专业 | 主要负责 |
|------|------|----------|
| 队员1 | 计算机科学 | 深度学习模型、训练pipeline、论文撰写 |
| 队员2 | 物理光电 | 荧光图像预处理、可解释性分析、原型系统 |

## 目标
- 学术论文（SCI）
- 可演示原型系统
- 项目研究报告
