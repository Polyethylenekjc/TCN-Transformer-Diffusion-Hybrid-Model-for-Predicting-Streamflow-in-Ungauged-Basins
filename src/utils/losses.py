import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, pred, target):
        """
        计算SSIM损失
        
        Args:
            pred: 预测图像
            target: 目标图像
            
        Returns:
            1-SSIM值作为损失
        """
        ssim_val = self._ssim(pred, target)
        return 1 - ssim_val

    def _ssim(self, img1, img2):
        """
        计算SSIM值
        """
        # 简化实现，实际应用中可以使用现成的SSIM实现
        return F.cosine_similarity(img1, img2, dim=-1).mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数
    """
    def __init__(self, lambda_diff=1.0, lambda_pix=0.1, lambda_perc=0.01, pix_loss_type='l1'):
        """
        Args:
            lambda_diff: 扩散损失权重
            lambda_pix: 像素损失权重
            lambda_perc: 感知损失权重
            pix_loss_type: 像素损失类型 ('l1' 或 'l2')
        """
        super(CombinedLoss, self).__init__()
        self.lambda_diff = lambda_diff
        self.lambda_pix = lambda_pix
        self.lambda_perc = lambda_perc
        
        self.diff_loss = DiffusionLoss()
        self.pix_loss = PixelLoss(pix_loss_type)
        self.perceptual_loss = PerceptualLoss()

    def forward(self, predicted_noise, true_noise, pred_image=None, target_image=None):
        """
        计算组合损失
        
        Args:
            predicted_noise: 预测的噪声
            true_noise: 真实的噪声
            pred_image: 预测图像（可选）
            target_image: 目标图像（可选）
            
        Returns:
            总损失值
        """
        # 扩散损失（主要损失）
        loss_diff = self.diff_loss(predicted_noise, true_noise)
        total_loss = self.lambda_diff * loss_diff
        
        # 如果提供了图像，则计算辅助损失
        if pred_image is not None and target_image is not None:
            # 像素损失
            loss_pix = self.pix_loss(pred_image, target_image)
            total_loss += self.lambda_pix * loss_pix
            
            # 感知损失
            loss_perc = self.perceptual_loss(pred_image, target_image)
            total_loss += self.lambda_perc * loss_perc
            
        return total_loss