# 肺癌分型深度学习项目 - 使用指南

## 项目概述

本项目基于MedicalNet的3D ResNet架构，实现肺癌病理类型的自动分类。支持多种类别不平衡处理策略，包括加权损失和批次平衡采样。

## 核心文件说明

### 1. 数据处理
- `dataset.py`: 完整的PyTorch数据管道，包含DataManager、Dataset类和两种DataLoader创建方法
- `util.py`: 工具函数（中心缩写、PID格式化、配置验证等）
- `config.yaml`: 项目配置文件

### 2. 模型架构  
- `model_classification.py`: 基于MedicalNet的3D ResNet分类模型（支持ResNet10/18/34等）

### 3. 训练框架
- `train.ipynb`: 使用加权损失策略的训练Notebook
- `train_balance.ipynb`: 使用批次平衡策略的训练Notebook
- 两种策略提供了不同的类别不平衡解决方案

## 快速开始

### 1. 环境准备
```bash
# 安装依赖（推荐使用conda环境）
pip install torch torchvision matplotlib seaborn
pip install scikit-learn pandas numpy yaml h5py tqdm
pip install monai  # 用于数据增强

# 检查数据文件结构
ls dataset/  # 应包含 manifest.csv, split.json, task.json, h5file/
```

### 2. 数据集测试
```python
# 测试数据加载
python dataset.py
```

### 3. 开始训练

有两种训练策略可选：

#### 策略1: 加权损失（推荐用于严重不平衡）
```bash
# 打开 train.ipynb 并运行所有单元格
# 使用加权CrossEntropyLoss处理类别不平衡
```

#### 策略2: 批次平衡（推荐用于中等不平衡）
```bash
# 打开 train_balance.ipynb 并运行所有单元格  
# 使用WeightedRandomSampler在批次级别平衡类别
```

## 配置文件说明

`config.yaml` 包含以下主要配置：

```yaml
# 数据路径配置
paths:
  data_root: "dataset/h5file"
  subjects_csv: "dataset/manifest.csv"
  splits_json: "dataset/split.json"
  tasks_json: "dataset/task.json"

# 实验配置
experiment:
  name: "baseline_pet_2class"
  output_dir: "training_results/baseline_pet_2class"
  seed: 42

# 数据配置
data:
  task_name: "ADC_vs_SCC"       # 对应task.json中的任务
  fold: "fold_0"                # 5-fold交叉验证的折数
  modalities: ["PET"]           # 使用的模态
  dtype: "float32"              # 数据类型
  split_train: "train"          # 训练集划分键名
  split_val: "val"              # 验证集划分键名

# 模型配置
model:
  architecture: "resnet18"      # 可选: resnet10, resnet18, resnet34
  num_classes: 2                # ADC vs SCC = 2类
  dropout_rate: 0.5
  pretrained_path: null         # 可指定MedicalNet预训练权重

# 训练配置
training:
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_epochs: 50
  patience: 10                  # 早停策略等待轮数
  save_interval: 10             # 检查点保存间隔

# 数据加载器配置
dataloader:
  train:
    batch_size: 8
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: false
  val:
    batch_size: 8
    shuffle: false
    num_workers: 4
    pin_memory: true
    drop_last: false
```

## 主要特性

### 1. 数据集特性
- ✅ **智能数据管理**: DataManager类集中处理所有元数据，避免重复I/O
- ✅ **多模态支持**: 支持PET、CT等多模态输入（配置中指定）
- ✅ **任务驱动筛选**: 根据task.json自动筛选对应任务的样本
- ✅ **交叉验证支持**: 支持5-fold交叉验证，配置文件指定fold
- ✅ **数据增强**: 集成MONAI数据增强（训练时随机3D旋转）
- ✅ **数据验证**: 内置完整性检查和错误处理

### 2. 模型特性
- ✅ **3D ResNet架构**: 专为医学3D图像设计的卷积网络
- ✅ **多种深度选择**: 支持ResNet10/18/34等不同深度
- ✅ **自适应输入**: 根据模态数量自动调整输入通道数
- ✅ **Dropout正则化**: 可配置的dropout防止过拟合

### 3. 训练特性
- ✅ **双重平衡策略**: 
  - 加权损失策略（train.ipynb）
  - 批次平衡策略（train_balance.ipynb）
- ✅ **ROC AUC监控**: 使用AUC作为主要评估指标和早停依据
- ✅ **学习率调度**: Warmup + 余弦退火调度器
- ✅ **自动保存**: 最佳模型、检查点、训练历史完整保存
- ✅ **可视化输出**: 训练曲线、ROC曲线、混淆矩阵自动生成
## 类别不平衡处理策略

项目提供了两种处理类别不平衡的策略：

### 策略1: 加权损失（train.ipynb）
```python
# 计算类别权重
class_weights = [total_samples / (num_classes * count) for count in train_label_counts]
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# 使用标准数据加载器
train_loader, val_loader, data_manager = create_dataloaders(config)
```

**特点：**
- 在损失函数层面平衡类别
- 适用于严重的类别不平衡
- 保持原始数据分布

### 策略2: 批次平衡（train_balance.ipynb）
```python
# 使用加权随机采样器
criterion = nn.CrossEntropyLoss()  # 标准损失函数

# 使用平衡数据加载器
train_loader, val_loader, data_manager = create_balanced_dataloaders(config)
```

**特点：**
- 在采样层面平衡类别
- 每个批次中类别分布更均衡
- 适用于中等程度的类别不平衡

### 选择建议
- **严重不平衡（比例 > 10:1）**: 推荐使用加权损失策略
- **中等不平衡（比例 2:1 - 10:1）**: 推荐使用批次平衡策略
- **轻微不平衡（比例 < 2:1）**: 两种策略效果相近

## 数据流程

```
原始数据 → DataManager → 任务筛选 → fold划分 → H5加载 → 数据增强 → 模型训练
```

1. **DataManager初始化**: 加载manifest.csv、split.json、task.json等元数据
2. **任务筛选**: 根据task.json中的labels_to_include筛选相关样本
3. **fold划分**: 根据split.json中的fold信息划分训练/验证集
4. **标签映射**: 将病理类型映射为数值标签（如ADC→0, SCC→1）
5. **H5数据加载**: 从h5file目录加载对应的3D医学图像数据
6. **数据增强**: 训练时应用随机3D旋转等增强技术
7. **批次采样**: 根据选择的策略进行标准采样或平衡采样
8. **模型训练**: 3D ResNet进行分类预测和参数优化

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

训练过程中会自动记录和可视化以下指标：

### 训练监控
- **ROC AUC**: 主要评估指标，用于早停和最佳模型选择
- **损失曲线**: 训练和验证损失的变化趋势
- **学习率调度**: Warmup + 余弦退火的学习率变化

### 最终评估
- **分类报告**: 每个类别的精确率、召回率、F1分数
- **混淆矩阵**: 预测与真实标签的对比矩阵
- **ROC曲线**: 真正率vs假正率曲线和AUC值
- **PR曲线**: 精确率-召回率曲线和平均精确率

### 输出文件
```
training_results/baseline_pet_2class/
├── best_model.pth              # 最佳模型权重
├── training_curves.png         # 训练曲线图
├── roc_curves.png             # ROC和PR曲线图  
├── confusion_matrix.png        # 混淆矩阵图
├── training_history.json       # 完整训练历史
└── checkpoint_epoch_*.pth      # 定期检查点
```

## 使用预训练权重

如果有MedicalNet预训练权重，可以在配置文件中指定：

```yaml
model:
  pretrained_path: "pretrain/resnet_18_23dataset.pth"
```

预训练权重的处理特点：
- 自动跳过不匹配的层（如分类头）
- 处理模块名前缀差异
- 自动报告加载成功的层数和跳过的层

## 数据格式要求

### H5文件格式
每个H5文件应包含：
```python
# H5文件结构
{
    'pet_data': np.array,  # Shape: [W, H, D] 
    'ct_data': np.array,   # Shape: [W, H, D] (可选)
    # 其他元数据...
}
```

### 元数据文件
- **manifest.csv**: 包含PID, center, pathology等字段
- **split.json**: 包含fold_0到fold_4的train/val划分
- **task.json**: 包含任务定义和标签映射

```json
{
  "ADC_vs_SCC": {
    "labels_to_include": ["ADC", "SCC"],
    "label_map": {"ADC": 0, "SCC": 1}
  }
}
```

## 扩展功能

### 多模态训练
```yaml
data:
  modalities: ["PET", "CT"]  # 使用PET+CT双模态

# 模型会自动设置 in_channels=2
```

### 多任务支持
```yaml
data:
  task_name: "NSCLC_vs_SCLC"  # 切换到其他任务

model:
  num_classes: 2  # 根据任务调整类别数
```

### 交叉验证
```yaml
data:
  fold: "fold_1"  # 使用不同的交叉验证fold (fold_0 到 fold_4)
```

### 不同模型架构
```yaml
model:
  architecture: "resnet34"  # 可选: resnet10, resnet18, resnet34
  dropout_rate: 0.3         # 调整正则化强度
```

## 故障排除

### 1. 数据加载错误
检查数据文件完整性和路径配置：
```python
# 验证配置文件路径
from dataset import load_config
from util import validate_config_paths

config = load_config("config.yaml")
validate_config_paths(config)  # 检查所有路径是否存在

# 测试H5文件
import h5py
h5_file = "dataset/h5file/AKH_xxx.h5"
with h5py.File(h5_file, 'r') as f:
    print("Keys:", list(f.keys()))
    print("PET shape:", f['pet_data'].shape)
```

### 2. 内存不足
减少batch_size或启用混合精度训练：
```yaml
dataloader:
  train:
    batch_size: 4  # 减少批次大小
  val:
    batch_size: 4
```

### 3. 类别不平衡严重
如果类别极度不平衡，可以尝试：
- 调整类别权重计算方式
- 使用focal loss替代CrossEntropyLoss
- 结合两种平衡策略

### 4. 训练不收敛
检查学习率和权重衰减设置：
```yaml
training:
  learning_rate: 0.00001  # 降低学习率
  weight_decay: 0.0001    # 增加正则化
```

### 5. GPU内存溢出
```bash
# 设置GPU内存增长
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 或者使用CPU训练（较慢）
device = torch.device("cpu")
```

## 项目结构
```
Lung_Cancer_Subtyping/
├── dataset.py                  # 完整数据管道实现
│   ├── DataManager             # 元数据管理器
│   ├── LungCancerDataset       # PyTorch Dataset类
│   ├── create_dataloaders()    # 标准数据加载器
│   └── create_balanced_dataloaders()  # 平衡数据加载器
├── util.py                     # 工具函数
├── model_classification.py     # 3D ResNet分类模型
├── config.yaml                 # 项目配置文件
├── train.ipynb                 # 加权损失训练脚本
├── train_balance.ipynb         # 批次平衡训练脚本
├── dataset/
│   ├── manifest.csv            # 样本元数据
│   ├── split.json              # 5-fold交叉验证划分
│   ├── task.json               # 任务定义和标签映射
│   └── h5file/                 # H5格式的3D医学图像
├── pretrain/                   # 预训练模型权重
│   ├── resnet_18.pth
│   └── resnet_18_23dataset.pth
└── training_results/           # 训练输出目录
    ├── baseline_pet_2class/    # 加权损失结果
    ├── baseline_pet_2class_batch_balance/  # 批次平衡结果
    │   ├── best_model.pth      # 最佳模型权重
    │   ├── training_curves.png # 训练曲线
    │   ├── roc_curves.png      # ROC和PR曲线
    │   ├── confusion_matrix.png # 混淆矩阵
    │   └── training_history.json # 训练历史
    └── ...
```

## 开发指南

### 添加新的平衡策略
1. 在`dataset.py`中添加新的DataLoader创建函数
2. 在训练脚本中调用新函数
3. 更新损失函数或采样器设置

### 添加新的模型架构
1. 在`model_classification.py`中实现新的分类器函数
2. 在配置文件中添加新的architecture选项
3. 更新模型创建逻辑

### 添加新的数据增强
1. 在`dataset.py`的`get_transforms()`函数中添加新的MONAI变换
2. 根据需要区分训练和验证时的增强策略

现在您可以开始使用这个完整的肺癌分型深度学习框架了！该框架提供了灵活的类别不平衡处理方案，完善的数据管道，以及全面的训练监控和评估功能。
