# 肺癌分型深度学习项目 - 使用指南

## 项目概述

本项目基于MedicalNet的3D ResNet架构，实现肺癌病理类型的自动分类（ADC vs SCC）。

## 核心文件说明

### 1. 数据处理
- `dataset.py`: PyTorch Dataset类，包含PID筛选逻辑
- `util.py`: 工具函数（中心缩写、PID格式化等）
- `config_clean.yaml`: 干净的配置文件

### 2. 模型架构  
- `model_classification.py`: 基于MedicalNet的分类模型
- `model_example.py`: 模型创建和使用示例
- `model`: 原始MedicalNet分割模型（参考用）

### 3. 训练框架
- `train_example.py`: 完整的训练脚本
- `test_integration.py`: 集成测试脚本

## 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 检查数据文件
ls dataset/  # 应包含 manifest.csv, split.json, task.json, h5file/
```

### 2. 测试集成
```bash
# 测试数据集和模型集成
python test_integration.py
```

### 3. 开始训练
```bash
# 使用默认配置训练
python train_example.py

# 或指定自定义配置
python -c "
from train_example import LungCancerTrainer
from dataset import load_config
config = load_config('config_clean.yaml')
trainer = LungCancerTrainer(config)
trainer.train(num_epochs=10)  # 快速测试
"
```

## 配置文件说明

`config_clean.yaml` 包含以下主要配置：

```yaml
# 实验配置
experiment:
  name: "baseline_pet_2class"
  output_dir: "baseline_pet_2class"

# 数据配置
data:
  task_name: "ADC_vs_SCC"       # 对应task.json中的任务
  fold: "fold_0"                # 5-fold交叉验证的折数
  modalities: ["PET"]           # 使用的模态

# 模型配置
model:
  architecture: "resnet18"      # 可选: resnet10, resnet18, resnet34, resnet50
  num_classes: 2                # ADC vs SCC = 2类
  dropout_rate: 0.5
  pretrained_path: null         # 可指定MedicalNet预训练权重

# 训练配置
training:
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_epochs: 50
```

## 主要特性

### 1. 数据集特性
- ✅ **智能PID筛选**: 根据任务标签和fold自动筛选样本
- ✅ **多模态支持**: 支持PET、CT等多模态输入
- ✅ **标签映射**: 自动处理病理类型到数值标签的映射
- ✅ **数据验证**: 内置数据完整性检查

### 2. 模型特性
- ✅ **3D ResNet**: 专为医学图像设计的3D卷积网络
- ✅ **预训练支持**: 支持加载MedicalNet预训练权重
- ✅ **多架构**: 支持ResNet10/18/34/50等不同深度
- ✅ **特征提取**: 可单独提取特征用于其他任务

### 3. 训练特性
- ✅ **自动保存**: 自动保存最佳模型和训练历史
- ✅ **学习率调度**: 内置ReduceLROnPlateau调度器
- ✅ **评估指标**: 准确率、分类报告、混淆矩阵
- ✅ **检查点**: 支持训练中断和恢复

## 数据流程

```
原始数据 → manifest.csv → 任务筛选 → fold划分 → H5加载 → 模型训练
```

1. **manifest.csv**: 包含所有样本的元数据（PID, center, pathology, fold_id）
2. **任务筛选**: 根据task.json中的labels_to_include筛选样本
3. **fold划分**: 根据split.json中的fold信息划分训练/验证集
4. **H5加载**: 从h5file目录加载对应的PET/CT数据
5. **模型训练**: 3D ResNet进行分类预测

## 模型架构

```
输入 [B, 1, 96, 64, 128] (PET数据)
    ↓
Conv3d(7×7×7) + BN + ReLU + MaxPool
    ↓
ResNet Layer1 (64 channels)
    ↓  
ResNet Layer2 (128 channels, stride=2)
    ↓
ResNet Layer3 (256 channels, stride=2)  
    ↓
ResNet Layer4 (512 channels, stride=2)
    ↓
AdaptiveAvgPool3d(1×1×1) + Dropout
    ↓
Linear(512 → 2) → [ADC, SCC]
```

## 性能指标

模型训练过程中会记录：
- 训练/验证损失和准确率
- 分类报告（精确率、召回率、F1分数）
- 混淆矩阵
- 各类别的性能指标

## 使用预训练权重

如果有MedicalNet预训练权重，可以这样使用：

```yaml
model:
  pretrained_path: "path/to/medicalnet_resnet_18.pth"
```

预训练权重加载器会自动：
- 跳过不匹配的层（如分类头）
- 处理模块名前缀差异
- 报告加载的层数

## 扩展功能

### 多模态训练
```yaml
data:
  modalities: ["PET", "CT"]  # 使用PET+CT双模态

model:
  # 会自动设置 in_channels=2
```

### 不同任务
```yaml
data:
  task_name: "NSCLC_vs_SCLC"  # 切换到其他任务
```

### 不同fold
```yaml
data:
  fold: "fold_1"  # 使用不同的交叉验证fold
```

## 故障排除

### 1. PyTorch符号错误
如果遇到 `undefined symbol: iJIT_NotifyEvent` 错误：
```bash
conda update intel-openmp mkl
# 或者
export OMP_NUM_THREADS=1
```

### 2. 数据加载错误
检查文件路径和H5文件完整性：
```python
import h5py
h5_file = "dataset/h5file/AKH_xxx.h5"
with h5py.File(h5_file, 'r') as f:
    print(list(f.keys()))
    print(f.attrs['pathology'])
```

### 3. 内存不足
减少batch_size或使用梯度累积：
```yaml
dataloader:
  train:
    batch_size: 4  # 减少批次大小
```

## 项目结构
```
Lung_Cancer_Subtyping/
├── dataset.py              # 数据集类
├── util.py                 # 工具函数  
├── model_classification.py # 分类模型
├── train_example.py        # 训练脚本
├── config_clean.yaml       # 配置文件
├── dataset/
│   ├── manifest.csv        # 样本元数据
│   ├── split.json          # 交叉验证划分
│   ├── task.json           # 任务定义
│   └── h5file/             # H5数据文件
└── baseline_pet_2class/    # 训练输出目录
    ├── best_model.pth      # 最佳模型
    ├── training_history.json
    └── evaluation_results.json
```

现在您可以开始使用这个完整的肺癌分型深度学习框架了！
