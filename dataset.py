# dataset.py
# 基础 PyTorch Dataset 类实现

import os
import h5py
import json
import pandas as pd
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Tuple, Optional, Any, Set
from util import get_center_abbreviation, get_pid_format

# MONAI imports for data augmentation
from monai.transforms import Compose, RandRotated


class DataManager:
    """
    数据管理器类 - 负责所有元数据管理和数据准备
    
    该类集中处理所有元数据的加载和过滤，避免重复的文件I/O操作
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据管理器
        
        Args:
            config: YAML配置字典
        """
        self.config = config
        self.data_root = config["paths"]["data_root"]
        self.task_name = config["data"]["task_name"]
        self.fold = config["data"]["fold"]
        self.modalities = config["data"]["modalities"]
        self.dtype = config["data"]["dtype"]
        
        # 加载所有必要的元数据
        self._load_metadata()
        
        # 处理数据过滤和分割
        self._process_data()
        
        # 验证数据完整性
        assert len(self.train_pids) == len(self.train_labels), \
            f"训练PID和标签数量不匹配: {len(self.train_pids)} vs {len(self.train_labels)}"
        assert len(self.val_pids) == len(self.val_labels), \
            f"验证PID和标签数量不匹配: {len(self.val_pids)} vs {len(self.val_labels)}"
        
        print(f"DataManager初始化完成:")
        print(f"- 任务: {self.task_name}")
        print(f"- Fold: {self.fold}")
        print(f"- 包含标签: {self.labels_to_include}")
        print(f"- 训练样本数: {len(self.train_pids)}")
        print(f"- 验证样本数: {len(self.val_pids)}")
        print(f"- 训练标签分布: {self.train_class_counts}")
        print(f"- 训练样本权重数量: {len(self.train_sample_weights)}")
    
    def _load_metadata(self):
        """加载所有元数据文件"""
        # 加载任务配置
        tasks_json_path = self.config["paths"]["tasks_json"]
        try:
            with open(tasks_json_path, 'r', encoding='utf-8') as f:
                tasks_config = json.load(f)
            
            if self.task_name not in tasks_config:
                raise ValueError(f"任务 '{self.task_name}' 不存在于 {tasks_json_path} 中")
            
            task_config = tasks_config[self.task_name]
            self.labels_to_include = task_config["labels_to_include"]
            self.label_map = task_config["label_map"]
            
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到任务配置文件: {tasks_json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"任务配置文件格式错误: {str(e)}")
        
        # 加载样本信息
        subjects_csv_path = self.config["paths"]["subjects_csv"]
        try:
            self.subjects_df = pd.read_csv(subjects_csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到样本文件: {subjects_csv_path}")
        
        # 加载数据分割信息
        splits_json_path = self.config["paths"]["splits_json"]
        try:
            with open(splits_json_path, 'r', encoding='utf-8') as f:
                self.splits_config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到数据划分文件: {splits_json_path}")
        
        if self.fold not in self.splits_config:
            raise ValueError(f"Fold '{self.fold}' 不存在于 {splits_json_path} 中")
    
    def _process_data(self):
        """处理数据过滤和分割"""
        # 1. 初始筛选：筛选出符合任务标签的样本
        task_mask = self.subjects_df['pathology'].isin(self.labels_to_include)
        task_subjects_df = self.subjects_df[task_mask].copy()
        
        # 2. 生成 combined_pid 列
        task_subjects_df['combined_pid'] = task_subjects_df.apply(
            lambda row: get_pid_format(row['center'], row['PID']), axis=1
        )
        
        # 3. 获取当前fold的训练和验证PID，用于后续筛选
        fold_config = self.splits_config[self.fold]
        split_train_key = self.config["data"]["split_train"]
        split_val_key = self.config["data"]["split_val"]

        fold_train_pids_set = set(fold_config.get(split_train_key, []))
        fold_val_pids_set = set(fold_config.get(split_val_key, []))

        # 4. 筛选出最终的训练和验证的 DataFrame
        train_mask = task_subjects_df['combined_pid'].isin(fold_train_pids_set)
        val_mask = task_subjects_df['combined_pid'].isin(fold_val_pids_set)

        train_df = task_subjects_df[train_mask].sort_values(by='combined_pid').reset_index(drop=True)
        val_df = task_subjects_df[val_mask].sort_values(by='combined_pid').reset_index(drop=True)
        
        # 5. 从最终的DataFrame中派生出所有需要的属性
        self.train_pids = train_df['combined_pid'].tolist()
        self.val_pids = val_df['combined_pid'].tolist()
        # 完成label map
        self.train_labels = train_df['pathology'].map(self.label_map).tolist()
        self.val_labels = val_df['pathology'].map(self.label_map).tolist()
        
        self.train_class_counts = train_df['pathology'].value_counts().to_dict()
        
        # 计算训练样本权重 (用于平衡采样)
        # 创建按类别索引排序的权重列表: 权重 = 1.0 / 该类别的样本数
        from collections import Counter
        label_counts = Counter(self.train_labels)
        class_weights_by_index = [1.0 / label_counts[i] for i in range(len(self.label_map))]
        # 为每个训练样本分配对应的权重
        self.train_sample_weights = [class_weights_by_index[label] for label in self.train_labels]
        
        # 创建完整的PID到pathology映射 (用于可能的外部查询)
        self.pid_to_pathology = dict(zip(task_subjects_df['combined_pid'], task_subjects_df['pathology']))
        
        # 检查样本数量
        if len(self.train_pids) == 0:
            raise ValueError(f"任务 {self.task_name} 在 {self.fold} train 下没有找到任何样本")
        if len(self.val_pids) == 0:
            raise ValueError(f"任务 {self.task_name} 在 {self.fold} val 下没有找到任何样本")


class LungCancerDataset(Dataset):
    
    def __init__(self, data_root: str, pids: List[str], labels: List[int], 
                 modalities: List[str], dtype: str, transforms: Optional[Compose] = None):
        """
        初始化数据集
        
        Args:
            data_root: H5文件根目录
            pids: PID列表
            labels: 对应的标签列表
            modalities: 模态列表
            dtype: 数据类型
            transforms: 数据增强变换
        """
        self.data_root = data_root
        self.pids = pids
        self.labels = labels
        self.modalities = modalities
        self.dtype = dtype
        self.transforms = transforms
        
        # 验证输入数据的一致性
        assert len(self.pids) == len(self.labels), \
            f"PID和标签数量不匹配: {len(self.pids)} vs {len(self.labels)}"
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.pids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (数据张量, 标签)
        """
        pid = self.pids[idx]
        label = self.labels[idx]
        
        try:
            # 构建H5文件路径
            h5_filename = f"{pid}.h5"
            h5_filepath = os.path.join(self.data_root, h5_filename)
            
            if not os.path.exists(h5_filepath):
                raise FileNotFoundError(f"H5文件不存在: {h5_filepath}")
            
            # 从H5文件加载数据
            with h5py.File(h5_filepath, 'r') as h5f:
                # 加载指定模态的数据
                data_list = []
                for modality in self.modalities:
                    if modality == "PET":
                        modal_data = h5f['pet_data'][:]
                    elif modality == "CT":
                        modal_data = h5f['ct_data'][:]
                    else:
                        raise ValueError(f"不支持的模态: {modality}")
                    
                    # H5数据原始维度是WHD，需要转换为DHW
                    # WHD -> DHW: transpose(2, 1, 0)
                    modal_data = modal_data.transpose(2, 1, 0)
                    
                    # 转换数据类型
                    if self.dtype == "float32":
                        modal_data = modal_data.astype(np.float32)
                    elif self.dtype == "float16":
                        modal_data = modal_data.astype(np.float16)
                    
                    data_list.append(modal_data)
            
            # 堆叠多模态数据 [C, D, H, W]
            if len(data_list) == 1:
                data_tensor = torch.from_numpy(data_list[0]).unsqueeze(0)  # 添加通道维度: [1, D, H, W]
            else:
                data_tensor = torch.from_numpy(np.stack(data_list, axis=0))  # [C, D, H, W]
            
            # 应用数据增强（如果提供）
            if self.transforms is not None:
                # 将数据转换为MONAI字典格式
                data_dict = {"image": data_tensor}
                # 应用变换
                data_dict = self.transforms(data_dict)  
                # 获取变换后的数据
                data_tensor = data_dict["image"]
            
            return data_tensor, label
            
        except Exception as e:
            print(f"加载样本 {pid} 失败: {str(e)}")
            # 返回零张量和对应标签 - 使用CDHW格式
            dummy_shape = (len(self.modalities), 128, 64, 96)  # [C, D, H, W] 格式
            dummy_tensor = torch.zeros(dummy_shape, dtype=torch.float32)
            return dummy_tensor, label

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"成功加载配置文件: {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式错误: {str(e)}")


def get_transforms(split: str) -> Optional[Compose]:
    """设置数据增强变换"""
    if split == "train":
        # 训练时使用旋转增强
        return Compose([
            RandRotated(
                keys=["image"],
                range_x=np.pi/12,  # ±15度
                range_y=np.pi/12,
                range_z=np.pi/12,
                prob=0.5,
                mode="bilinear"
            )
        ])
    else:
        # 验证时不使用增强
        return None


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataManager]:
    """
    创建训练和验证数据加载器
    
    Args:
        config: YAML配置字典
        
    Returns:
        tuple: (训练数据加载器, 验证数据加载器, 数据管理器)
    """
    
    # 创建数据管理器 - 只需实例化一次
    data_manager = DataManager(config)
    
    # 设置数据增强变换
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")
    
    # 创建数据集实例
    train_dataset = LungCancerDataset(
        data_root=data_manager.data_root,
        pids=data_manager.train_pids,
        labels=data_manager.train_labels,
        modalities=data_manager.modalities,
        dtype=data_manager.dtype,
        transforms=train_transforms
    )
    
    val_dataset = LungCancerDataset(
        data_root=data_manager.data_root,
        pids=data_manager.val_pids,
        labels=data_manager.val_labels,
        modalities=data_manager.modalities,
        dtype=data_manager.dtype,
        transforms=val_transforms
    )
    
    # 获取数据加载器配置
    train_config = config["dataloader"]["train"]
    val_config = config["dataloader"]["val"]
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=train_config["shuffle"],
        num_workers=train_config["num_workers"],
        pin_memory=train_config["pin_memory"],
        drop_last=train_config["drop_last"]
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_config["batch_size"],
        shuffle=val_config["shuffle"],
        num_workers=val_config["num_workers"],
        pin_memory=val_config["pin_memory"],
        drop_last=val_config["drop_last"]
    )
    
    print(f"\n数据加载器创建完成:")
    print(f"- 训练数据增强: {'启用' if train_transforms else '禁用'}")
    print(f"- 验证数据增强: {'启用' if val_transforms else '禁用'}")
    
    return train_dataloader, val_dataloader, data_manager


def create_balanced_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataManager]:
    """
    创建平衡的训练和验证数据加载器
    
    使用WeightedRandomSampler对训练集进行平衡采样，以处理类别不平衡问题。
    
    Args:
        config: YAML配置字典
        
    Returns:
        tuple: (平衡的训练数据加载器, 验证数据加载器, 数据管理器)
    """
    
    # 创建数据管理器 - 只需实例化一次
    data_manager = DataManager(config)
    
    # 设置数据增强变换
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")
    
    # 创建数据集实例
    train_dataset = LungCancerDataset(
        data_root=data_manager.data_root,
        pids=data_manager.train_pids,
        labels=data_manager.train_labels,
        modalities=data_manager.modalities,
        dtype=data_manager.dtype,
        transforms=train_transforms
    )
    
    val_dataset = LungCancerDataset(
        data_root=data_manager.data_root,
        pids=data_manager.val_pids,
        labels=data_manager.val_labels,
        modalities=data_manager.modalities,
        dtype=data_manager.dtype,
        transforms=val_transforms
    )
    
    # 获取数据加载器配置
    train_config = config["dataloader"]["train"]
    val_config = config["dataloader"]["val"]
    
    # 创建加权随机采样器用于训练集平衡
    train_sampler = WeightedRandomSampler(
        weights=data_manager.train_sample_weights,
        num_samples=len(data_manager.train_sample_weights),
        replacement=True  # 允许重复采样以实现平衡
    )
    
    # 创建平衡的训练数据加载器
    # 注意：使用sampler时必须设置shuffle=False
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        sampler=train_sampler,  # 使用加权采样器
        shuffle=False,          # 采样器处理随机化，必须设为False
        num_workers=train_config["num_workers"],
        pin_memory=train_config["pin_memory"],
        drop_last=train_config["drop_last"]
    )
    
    # 验证数据加载器保持不变
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_config["batch_size"],
        shuffle=val_config["shuffle"],
        num_workers=val_config["num_workers"],
        pin_memory=val_config["pin_memory"],
        drop_last=val_config["drop_last"]
    )
    
    print(f"\n平衡数据加载器创建完成:")
    print(f"- 训练数据增强: {'启用' if train_transforms else '禁用'}")
    print(f"- 验证数据增强: {'启用' if val_transforms else '禁用'}")
    print(f"- 训练集采样器: {type(train_sampler).__name__}")
    print(f"- 采样权重数量: {len(data_manager.train_sample_weights)}")
    
    return train_dataloader, val_dataloader, data_manager


if __name__ == "__main__":
    # 测试代码
    print("测试重构后的Dataset类...")
    
    # 加载配置
    config = load_config("config.yaml")
    
    # 测试标准数据加载器
    print("\n" + "="*60)
    print("测试标准数据加载器")
    print("="*60)
    # train_loader, val_loader, data_manager = create_dataloaders(config)
    
    # 测试平衡数据加载器
    print("\n" + "="*60)
    print("测试平衡数据加载器")
    print("="*60)
    train_loader, val_loader, data_manager = create_balanced_dataloaders(config)
    
    # 显示数据管理器统计信息
    print(f"\n数据管理器统计:")
    print(f"- 标签映射: {data_manager.label_map}")
    print(f"- 训练类别分布: {data_manager.train_class_counts}")
    print(f"- 训练样本数: {len(data_manager.train_pids)}")
    print(f"- 验证样本数: {len(data_manager.val_pids)}")
    print(f"- 训练集采样器: {type(train_loader.sampler).__name__}")
    
    # 测试加载一个批次
    print("\n测试数据加载:")
    try:
        # 测试训练集
        train_iter = iter(train_loader)
        batch_data, batch_labels = next(train_iter)
        print(f"训练批次数据形状: {batch_data.shape}")
        print(f"训练批次标签形状: {batch_labels.shape}")
        print(f"训练标签样例: {batch_labels}")
        
        # 分析平衡效果 - 统计多个批次的标签分布
        print(f"\n平衡效果分析 (采样10个批次):")
        label_counts_total = {}
        num_test_batches = 10
        
        for i in range(num_test_batches):
            try:
                _, batch_labels = next(train_iter)
                for label in batch_labels.numpy():
                    label_counts_total[label] = label_counts_total.get(label, 0) + 1
            except StopIteration:
                train_iter = iter(train_loader)  # 重新开始
                _, batch_labels = next(train_iter)
                for label in batch_labels.numpy():
                    label_counts_total[label] = label_counts_total.get(label, 0) + 1
        
        total_samples = sum(label_counts_total.values())
        print(f"采样结果 ({total_samples} 个样本):")
        for label, count in sorted(label_counts_total.items()):
            percentage = (count / total_samples) * 100
            print(f"  标签 {label}: {count} 样本 ({percentage:.1f}%)")
        
        # 测试验证集
        val_iter = iter(val_loader)
        batch_data, batch_labels = next(val_iter)
        print(f"\n验证批次数据形状: {batch_data.shape}")
        print(f"验证批次标签形状: {batch_labels.shape}")
        print(f"验证标签样例: {batch_labels}")
        
        # 验证标签范围
        train_labels_set = set(data_manager.train_labels)
        val_labels_set = set(data_manager.val_labels)
        print(f"\n训练集标签范围: {train_labels_set}")
        print(f"验证集标签范围: {val_labels_set}")
        
    except Exception as e:
        print(f"数据加载测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
