# 基于深度学习的口腔荧光图像辅助诊断
 
**北京大学口腔医院 | 本科生创新实践项目**
 
## 项目概述
 
针对北京大学口腔医院提供的口腔荧光图像临床数据集，采用深度学习**目标检测**方法
（YOLO 系列）实现口腔黏膜病变的自动检测与分类，并结合可视化技术提供可解释性
诊断依据。最终产出学术论文、软件程序与研究报告。
 
## 任务定位
 
- **任务类型**：目标检测（bounding box）
- **标注格式**：PASCAL VOC XML → 转为 YOLO txt
- **类别**：见 `configs/dataset.yaml`（`l` / `n`，含义待与标注方确认）
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
 
## 团队分工
 
| 角色 | 专业 | 主要负责 |
|------|------|----------|
| 队员1 | 计算机科学 | 模型、训练 pipeline、论文 |
| 队员2 | 物理光电 | 荧光图像预处理、可解释性、原型系统 |
 
## 目标成果
 
- 学术论文（SCI）
- 可演示原型系统
- 项目研究报告
