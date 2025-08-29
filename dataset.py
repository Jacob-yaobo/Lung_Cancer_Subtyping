# dataset.py
# 基础 PyTorch Dataset 类实现

import os
import h5py
import json
import pandas as pd
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Set
from util import get_center_abbreviation, get_pid_format

# MONAI imports for data augmentation
from monai.transforms import Compose, RandRotated

class LungCancerDataset(Dataset):
    """
    肺癌分型数据集类
    
    根据配置文件加载指定任务的数据
    """
    
    def __init__(self, config: Dict[str, Any], split: str = "train"):
        """
        初始化数据集
        
        Args:
            config: YAML配置字典
            split: 数据集划分 ("train" 或 "val")
        """
        self.config = config
        self.split = split
        
        # 解析配置
        self.data_root = config["paths"]["data_root"]

        self.task_name = config["data"]["task_name"]
        self.fold = config["data"]["fold"] 
        self.modalities = config["data"]["modalities"]
        self.dtype = config["data"]["dtype"]
        
        # 根据split确定对应的PID键名
        if split == "train":
            self.split_key = config["data"]["split_train"]  # "train_pids"
        elif split == "val":
            self.split_key = config["data"]["split_val"]    # "val_pids"
        else:
            raise ValueError(f"不支持的split类型: {split}")
        
        # 加载任务配置
        self.task_config = self._load_task_config()
        self.labels_to_include = self.task_config["labels_to_include"]
        self.label_map = self.task_config["label_map"]
        
        # 获取当前任务和split下的PID列表
        self.pid_list = self._get_task_split_pids()
        
        # 设置简单的数据增强
        self._setup_transforms()
        
        print(f"初始化 {split} 数据集:")
        print(f"- 任务: {self.task_name}")
        print(f"- Fold: {self.fold}")
        print(f"- 包含标签: {self.labels_to_include}")
        print(f"- 样本数量: {len(self.pid_list)}")
        print(f"- 数据增强: {'启用' if self.transforms else '禁用'}")
        # print(f"- 标签分布: {self._get_label_distribution()}")
    
    def _setup_transforms(self):
        """设置简单的数据增强 - 只有旋转"""
        if self.split == "train":
            # 训练时使用旋转增强
            self.transforms = Compose([
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
            self.transforms = None
    
    def _load_task_config(self) -> Dict[str, Any]:
        """从tasks.json加载任务配置"""
        tasks_json_path = self.config["paths"]["tasks_json"]
        
        try:
            with open(tasks_json_path, 'r', encoding='utf-8') as f:
                tasks_config = json.load(f)
            
            if self.task_name not in tasks_config:
                raise ValueError(f"任务 '{self.task_name}' 不存在于 {tasks_json_path} 中")
            
            return tasks_config[self.task_name]
            
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到任务配置文件: {tasks_json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"任务配置文件格式错误: {str(e)}")
    
    def _get_task_split_pids(self) -> List[str]:
        """
        获取当前任务和split下的PID列表
        
        流程:
        1. 从tasks.json获取labels_to_include
        2. 从subjects.csv根据pathology筛选出符合任务的PID
        3. 从splits.json获取当前fold和split的PID
        4. 取交集得到最终的PID列表
        """
        
        # 1. 从subjects.csv加载所有样本
        subjects_csv_path = self.config["paths"]["subjects_csv"]
        try:
            df = pd.read_csv(subjects_csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到样本文件: {subjects_csv_path}")
        
        # 2. 从CSV文件中，根据pathology筛选符合当前任务的PID
        task_pids = set()
        for _, row in df.iterrows():
            if row['pathology'] in self.labels_to_include:
                # 生成带中心缩写的PID格式 (与split.json中的格式一致)
                combined_pid = get_pid_format(row['center'], row['PID'])
                task_pids.add(combined_pid)
        
        # print(f"任务 {self.task_name} 包含的样本数: {len(task_pids)}")
        
        # 3. 从splits.json获取当前fold和split的PID
        splits_json_path = self.config["paths"]["splits_json"]
        try:
            with open(splits_json_path, 'r', encoding='utf-8') as f:
                splits_config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到数据划分文件: {splits_json_path}")
        
        if self.fold not in splits_config:
            raise ValueError(f"Fold '{self.fold}' 不存在于 {splits_json_path} 中")
        
        if self.split_key not in splits_config[self.fold]:
            raise ValueError(f"Split key '{self.split_key}' 不存在于 {self.fold} 中")
        
        split_pids = set(splits_config[self.fold][self.split_key])
        # print(f"Fold {self.fold} {self.split} 包含的样本数: {len(split_pids)}")
        
        # 4. 取交集得到最终的PID列表
        final_pids = list(task_pids.intersection(split_pids))
        # print(f"最终交集样本数: {len(final_pids)}")
        
        if len(final_pids) == 0:
            raise ValueError(f"任务 {self.task_name} 在 {self.fold} {self.split} 下没有找到任何样本")
        
        return final_pids
    
    def _get_label_distribution(self) -> Dict[str, int]:
        """获取标签分布"""
        # 加载subjects.csv获取pathology信息
        subjects_csv_path = self.config["paths"]["subjects_csv"]
        df = pd.read_csv(subjects_csv_path)
        
        # 创建PID到pathology的映射
        pid_to_pathology = {}
        for _, row in df.iterrows():
            combined_pid = get_pid_format(row['center'], row['PID'])
            pid_to_pathology[combined_pid] = row['pathology']
        
        # 统计当前PID列表的标签分布
        label_count = {}
        for pid in self.pid_list:
            if pid in pid_to_pathology:
                pathology = pid_to_pathology[pid]
                label_count[pathology] = label_count.get(pathology, 0) + 1
        
        return label_count
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.pid_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (数据张量, 标签)
        """
        combined_pid = self.pid_list[idx]
        
        try:
            # 构建H5文件路径
            h5_filename = f"{combined_pid}.h5"
            h5_filepath = os.path.join(self.data_root, h5_filename)
            
            if not os.path.exists(h5_filepath):
                raise FileNotFoundError(f"H5文件不存在: {h5_filepath}")
            
            # 从H5文件加载数据
            with h5py.File(h5_filepath, 'r') as h5f:
                # 获取pathology标签
                pathology = h5f.attrs['pathology']
                if isinstance(pathology, bytes):
                    pathology = pathology.decode('utf-8')
                
                # 检查标签是否在当前任务中
                if pathology not in self.labels_to_include:
                    raise ValueError(f"样本 {combined_pid} 的标签 {pathology} 不在任务 {self.task_name} 中")
                
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
            
            # 应用数据增强（仅在训练时）
            if self.transforms is not None:
                # 将数据转换为MONAI字典格式
                data_dict = {"image": data_tensor}
                # 应用变换
                data_dict = self.transforms(data_dict)  
                # 获取变换后的数据
                data_tensor = data_dict["image"]
            
            # 获取标签
            label = self.label_map[pathology]
            
            return data_tensor, label
            
        except Exception as e:
            print(f"加载样本 {combined_pid} 失败: {str(e)}")
            # 返回零张量和默认标签 - 使用CDHW格式
            dummy_shape = (len(self.modalities), 128, 64, 96)  # [C, D, H, W] 格式
            dummy_tensor = torch.zeros(dummy_shape, dtype=torch.float32)
            return dummy_tensor, 0

def load_config(config_path: str = "congfig.yaml") -> Dict[str, Any]:
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

def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    
    # 创建数据集
    train_dataset = LungCancerDataset(config, split="train")
    val_dataset = LungCancerDataset(config, split="val")
    
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
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    # 测试代码
    print("测试Dataset类...")
    
    # 加载配置
    config = load_config("congfig.yaml")
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(config)
    
    # 测试加载一个批次
    print("\n测试数据加载:")
    try:
        # 测试训练集
        train_iter = iter(train_loader)
        batch_data, batch_labels = next(train_iter)
        print(f"训练批次数据形状: {batch_data.shape}")
        print(f"训练批次标签形状: {batch_labels.shape}")
        print(f"训练标签样例: {batch_labels[:5]}")
        
        # 测试验证集
        val_iter = iter(val_loader)
        batch_data, batch_labels = next(val_iter)
        print(f"验证批次数据形状: {batch_data.shape}")
        print(f"验证批次标签形状: {batch_labels.shape}")
        print(f"验证标签样例: {batch_labels[:5]}")
        
    except Exception as e:
        print(f"数据加载测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
