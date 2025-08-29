# util.py
# 工具函数集合

from typing import Dict, Any

def get_center_abbreviation(center_name: str) -> str:
    """
    获取中心缩写
    
    Args:
        center_name: 中心全名
        
    Returns:
        str: 中心缩写
        
    Raises:
        ValueError: 当中心名称未知时
    """
    center_abbr_map = {
        'AKH_nifti_637': 'AKH',
        'Neimeng_nifti_425': 'Neimeng'
    }
    
    if center_name in center_abbr_map:
        return center_abbr_map[center_name]
    else:
        # 如果没有找到映射，尝试直接使用前缀
        if 'AKH' in center_name:
            return 'AKH'
        elif 'Neimeng' in center_name:
            return 'Neimeng'
        else:
            raise ValueError(f"未知的中心名称: {center_name}")

def validate_config_paths(config: Dict[str, Any]) -> None:
    """
    验证配置文件中的路径是否存在
    
    Args:
        config: 配置字典
        
    Raises:
        FileNotFoundError: 当必要文件不存在时
    """
    import os
    
    required_paths = [
        ("subjects_csv", "样本文件"),
        ("splits_json", "数据划分文件"), 
        ("tasks_json", "任务配置文件"),
        ("data_root", "数据根目录")
    ]
    
    for path_key, description in required_paths:
        if path_key in config["paths"]:
            path = config["paths"][path_key]
            if not os.path.exists(path):
                raise FileNotFoundError(f"{description}不存在: {path}")

def get_pid_format(center: str, pid: str) -> str:
    """
    生成统一的PID格式 (中心缩写_PID)
    
    Args:
        center: 中心名称
        pid: 患者ID
        
    Returns:
        str: 格式化的PID
    """
    center_abbr = get_center_abbreviation(center)
    return f"{center_abbr}_{pid}"

def parse_pid_format(combined_pid: str) -> tuple[str, str]:
    """
    解析组合PID格式，分离出中心缩写和患者ID
    
    Args:
        combined_pid: 格式为 "中心缩写_患者ID" 的字符串
        
    Returns:
        tuple: (中心缩写, 患者ID)
        
    Raises:
        ValueError: 当PID格式不正确时
    """
    parts = combined_pid.split('_', 1)
    if len(parts) != 2:
        raise ValueError(f"PID格式不正确: {combined_pid}，应为 '中心缩写_患者ID'")
    
    center_abbr, pid = parts
    return center_abbr, pid
