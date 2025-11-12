import torch
import torchvision.transforms as transforms
import numpy as np
import random


class DataPreprocessor:
    """
    数据预处理类
    """
    def __init__(self, normalize_method='z_score'):
        """
        Args:
            normalize_method: 归一化方法 ('z_score' 或 'min_max')
        """
        self.normalize_method = normalize_method
        self.stats = {}  # 存储各通道的统计信息

    def fit_stats(self, data_loader):
        """
        从数据集中计算统计数据
        
        Args:
            data_loader: 数据加载器
        """
        if self.normalize_method == 'z_score':
            self._compute_zscore_stats(data_loader)
        elif self.normalize_method == 'min_max':
            self._compute_minmax_stats(data_loader)

    def _compute_zscore_stats(self, data_loader):
        """
        计算z-score归一化所需的均值和标准差
        """
        # 这里只是一个示例实现，实际使用时应根据具体数据格式调整
        pass

    def _compute_minmax_stats(self, data_loader):
        """
        计算min-max归一化所需的最小值和最大值
        """
        # 这里只是一个示例实现，实际使用时应根据具体数据格式调整
        pass

    def normalize(self, data, channel_axis=1):
        """
        归一化数据
        
        Args:
            data: 输入数据
            channel_axis: 通道轴索引
            
        Returns:
            归一化后的数据
        """
        if self.normalize_method == 'z_score':
            return self._zscore_normalize(data, channel_axis)
        elif self.normalize_method == 'min_max':
            return self._minmax_normalize(data, channel_axis)
        else:
            return data

    def _zscore_normalize(self, data, channel_axis):
        """
        Z-score归一化
        """
        # 示例实现
        return data

    def _minmax_normalize(self, data, channel_axis):
        """
        Min-Max归一化
        """
        # 示例实现
        return data

    def denormalize(self, data, channel_axis=1):
        """
        反归一化数据
        
        Args:
            data: 归一化后的数据
            channel_axis: 通道轴索引
            
        Returns:
            原始范围的数据
        """
        # 示例实现
        return data


class DataAugmentation:
    """
    数据增强类
    """
    def __init__(self, augmentations=None):
        """
        Args:
            augmentations: 增强操作列表
        """
        if augmentations is None:
            self.augmentations = ['flip', 'rotate']
        else:
            self.augmentations = augmentations

    def apply_augmentation(self, x_seq, y_gt=None):
        """
        应用数据增强
        
        Args:
            x_seq: 输入序列 (B, T, C, H, W)
            y_gt: 真实标签 (B, 1, H_out, W_out)
            
        Returns:
            增强后的数据
        """
        # 随机选择是否进行增强
        if random.random() < 0.5:
            return x_seq, y_gt
            
        # 应用各种增强操作
        for aug in self.augmentations:
            if aug == 'flip':
                x_seq, y_gt = self._random_flip(x_seq, y_gt)
            elif aug == 'rotate':
                x_seq, y_gt = self._random_rotate(x_seq, y_gt)
            elif aug == 'crop':
                x_seq, y_gt = self._random_crop(x_seq, y_gt)
                
        return x_seq, y_gt

    def _random_flip(self, x_seq, y_gt):
        """
        随机翻转
        """
        if random.random() < 0.5:
            # 水平翻转
            x_seq = torch.flip(x_seq, [-1])
            if y_gt is not None:
                y_gt = torch.flip(y_gt, [-1])
                
        if random.random() < 0.5:
            # 垂直翻转
            x_seq = torch.flip(x_seq, [-2])
            if y_gt is not None:
                y_gt = torch.flip(y_gt, [-2])
                
        return x_seq, y_gt

    def _random_rotate(self, x_seq, y_gt):
        """
        随机旋转
        """
        if random.random() < 0.5:
            k = random.randint(1, 3)  # 90, 180, 或 270度旋转
            x_seq = torch.rot90(x_seq, k, [-2, -1])
            if y_gt is not None:
                y_gt = torch.rot90(y_gt, k, [-2, -1])
                
        return x_seq, y_gt

    def _random_crop(self, x_seq, y_gt):
        """
        随机裁剪
        """
        # 示例实现，实际使用时需要根据具体需求调整
        return x_seq, y_gt


def create_data_loader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 加载数据的工作进程数
        
    Returns:
        DataLoader对象
    """
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)