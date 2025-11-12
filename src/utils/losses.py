import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class DiffusionLoss(nn.Module):
    """
    扩散模型损失函数
    """
    def __init__(self):
        super(DiffusionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_noise, true_noise):
        """
        计算扩散损失
        
        Args:
            predicted_noise: 预测的噪声
            true_noise: 真实的噪声
            
        Returns:
            损失值
        """
        return self.mse_loss(predicted_noise, true_noise)


class PixelLoss(nn.Module):
    """
    像素级损失函数 (L1/L2)
    """
    def __init__(self, loss_type='l1'):
        super(PixelLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("loss_type must be 'l1' or 'l2'")

    def forward(self, pred, target):
        """
        计算像素级损失
        
        Args:
            pred: 预测图像
            target: 目标图像
            
        Returns:
            损失值
        """
        return self.loss_fn(pred, target)


class StationLoss(nn.Module):
    """
    基于观测站点数据的径流约束损失函数
    """
    def __init__(self, stations_csv, resolution=1.0, loss_type='l1'):
        """
        初始化站点损失函数
        
        Args:
            stations_csv: 包含站点信息的CSV文件路径，应包含列：lat, lon, runoff
            resolution: 输出图像的地理分辨率（度），默认为1.0度
            loss_type: 损失函数类型 ('l1' 或 'mse')
        """
        super(StationLoss, self).__init__()
        self.stations_df = pd.read_csv(stations_csv)
        self.resolution = resolution
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='mean')
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='mean')
        else:
            raise ValueError("loss_type must be 'l1' or 'mse'")
    
    def _get_pixel_coords(self, lat, lon, image_height, image_width, lat_min, lon_min):
        """
        将经纬度转换为图像像素坐标
        
        Args:
            lat: 纬度
            lon: 经度
            image_height: 图像高度
            image_width: 图像宽度
            lat_min: 图像最小纬度
            lon_min: 图像最小经度
            
        Returns:
            (row, col): 像素坐标，如果超出范围则返回(None, None)
        """
        # 计算相对于左下角的位置
        row = int((lat - lat_min) / self.resolution)
        col = int((lon - lon_min) / self.resolution)
        
        # 检查是否在图像范围内
        if 0 <= row < image_height and 0 <= col < image_width:
            return row, col
        else:
            return None, None
    
    def forward(self, pred_images, target_runoff_values, time_steps=None):
        """
        计算站点损失
        
        Args:
            pred_images: 预测图像 [B, 1, H, W]
            target_runoff_values: 目标径流值 [N_stations, N_timesteps] 或 [N_stations]
            time_steps: 时间步索引 [N_timesteps] 或 None
            
        Returns:
            站点损失值
        """
        batch_size, _, height, width = pred_images.shape
        total_loss = 0.0
        valid_station_count = 0
        
        # 获取图像的地理范围（假设图像覆盖从(0,0)开始的区域）
        lat_min, lon_min = 0.0, 0.0
        lat_max = lat_min + height * self.resolution
        lon_max = lon_min + width * self.resolution
        
        # 处理单个时间步的情况
        single_time_step = target_runoff_values.dim() == 1
        
        for idx, (_, station) in enumerate(self.stations_df.iterrows()):
            lat, lon = station['lat'], station['lon']
            
            # 获取像素坐标
            row, col = self._get_pixel_coords(lat, lon, height, width, lat_min, lon_min)
            
            # 如果站点在图像范围内
            if row is not None and col is not None:
                # 提取预测值（在站点位置的像素值）
                pred_value = pred_images[:, :, row, col]  # [B, 1]
                
                # 获取目标值
                if single_time_step:
                    # 单个时间步情况
                    target_values = target_runoff_values[idx].unsqueeze(0).expand(batch_size, 1)  # [B, 1]
                else:
                    # 多个时间步情况
                    if time_steps is not None:
                        target_values = target_runoff_values[idx, time_steps].unsqueeze(1)  # [B, 1]
                    else:
                        # 默认使用前batch_size个时间步
                        target_values = target_runoff_values[idx, :batch_size].unsqueeze(1)  # [B, 1]
                
                # 计算损失
                station_loss = self.loss_fn(pred_value, target_values)
                total_loss += station_loss
                valid_station_count += 1
        
        # 返回平均损失
        if valid_station_count > 0:
            return total_loss / valid_station_count
        else:
            # 如果没有有效站点，返回0
            return torch.tensor(0.0, device=pred_images.device)


class TotalLoss(nn.Module):
    """
    总损失函数，结合重建损失和站点损失
    """
    def __init__(self, stations_csv, resolution=1.0, lambda_recon=1.0, lambda_station=0.1, 
                 recon_loss_type='l1', station_loss_type='l1'):
        """
        初始化总损失函数
        
        Args:
            stations_csv: 包含站点信息的CSV文件路径
            resolution: 输出图像的地理分辨率（度）
            lambda_recon: 重建损失权重
            lambda_station: 站点损失权重
            recon_loss_type: 重建损失类型 ('l1', 'l2', 'ssim')
            station_loss_type: 站点损失类型 ('l1', 'mse')
        """
        super(TotalLoss, self).__init__()
        self.lambda_recon = lambda_recon
        self.lambda_station = lambda_station
        
        # 初始化重建损失
        if recon_loss_type in ['l1', 'l2']:
            self.recon_loss = PixelLoss(recon_loss_type)
        elif recon_loss_type == 'ssim':
            self.recon_loss = SSIMLoss()
        else:
            raise ValueError("recon_loss_type must be 'l1', 'l2', or 'ssim'")
        
        # 初始化站点损失
        self.station_loss = StationLoss(stations_csv, resolution, station_loss_type)
        
        # 初始化扩散损失
        self.diffusion_loss = DiffusionLoss()
    
    def forward(self, pred_images, target_images, target_runoff_values, time_steps=None):
        """
        计算总损失
        
        Args:
            pred_images: 预测图像 [B, 1, H, W]
            target_images: 目标图像 [B, 1, H, W]
            target_runoff_values: 目标径流值 [N_stations, N_timesteps] 或 [N_stations]
            time_steps: 时间步索引 [N_timesteps] 或 None
            
        Returns:
            总损失值，重建损失，站点损失
        """
        # 计算重建损失
        recon_loss = self.recon_loss(pred_images, target_images)
        
        # 计算站点损失
        station_loss = self.station_loss(pred_images, target_runoff_values, time_steps)
        
        # 计算总损失
        total_loss = self.lambda_recon * recon_loss + self.lambda_station * station_loss
        
        return total_loss, recon_loss, station_loss


class PerceptualLoss(nn.Module):
    """
    感知损失函数
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # 这里可以使用预训练的VGG网络计算感知损失
        # 为简化实现，这里只是示例
        self.loss_fn = nn.MSELoss()

    def forward(self, pred, target):
        """
        计算感知损失
        
        Args:
            pred: 预测图像
            target: 目标图像
            
        Returns:
            损失值
        """
        # 实际实现中应该使用VGG等网络提取特征再计算损失
        return self.loss_fn(pred, target)


class SSIMLoss(nn.Module):
    """
    SSIM损失函数
    """
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.gaussian_window = self._create_window(window_size, 1)

    def _create_window(self, window_size, channel):
        """
        创建高斯窗口
        
        Args:
            window_size: 窗口大小
            channel: 通道数
            
        Returns:
            高斯窗口
        """
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _gaussian(self, window_size, sigma):
        """
        生成一维高斯核
        
        Args:
            window_size: 窗口大小
            sigma: 高斯核标准差
            
        Returns:
            一维高斯核
        """
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def forward(self, img1, img2):
        """
        计算SSIM损失
        
        Args:
            img1: 图像1
            img2: 图像2
            
        Returns:
            SSIM损失值
        """
        (_, channel, _, _) = img1.size()
        
        if channel == self.gaussian_window.shape[0]:
            window = self.gaussian_window
        else:
            window = self._create_window(self.window_size, channel)
            
        window = window.to(img1.device).type_as(img1)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


# 为了向后兼容，保留CombinedLoss
class CombinedLoss(nn.Module):
    """
    结合多种损失函数的组合损失
    """
    def __init__(self, lambda_diff=1.0, lambda_pix=0.1, lambda_perc=0.01):
        super(CombinedLoss, self).__init__()
        self.lambda_diff = lambda_diff
        self.lambda_pix = lambda_pix
        self.lambda_perc = lambda_perc
        self.diff_loss = DiffusionLoss()
        self.pix_loss = PixelLoss()
        self.perc_loss = PerceptualLoss()

    def forward(self, predicted_noise, true_noise):
        """
        计算组合损失
        
        Args:
            predicted_noise: 预测噪声
            true_noise: 真实噪声
            
        Returns:
            组合损失值
        """
        diff_loss = self.diff_loss(predicted_noise, true_noise)
        # 注意：这里简化处理，实际应用中可能需要生成图像用于像素和感知损失计算
        pix_loss = torch.tensor(0.0, device=predicted_noise.device)
        perc_loss = torch.tensor(0.0, device=predicted_noise.device)
        
        combined_loss = self.lambda_diff * diff_loss + self.lambda_pix * pix_loss + self.lambda_perc * perc_loss
        return combined_loss