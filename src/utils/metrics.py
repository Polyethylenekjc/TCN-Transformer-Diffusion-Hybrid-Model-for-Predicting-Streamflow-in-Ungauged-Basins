import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def mse_metric(pred, target):
    """
    计算MSE指标
    
    Args:
        pred: 预测图像
        target: 目标图像
        
    Returns:
        MSE值
    """
    return F.mse_loss(pred, target).item()


def rmse_metric(pred, target):
    """
    计算RMSE指标
    
    Args:
        pred: 预测图像
        target: 目标图像
        
    Returns:
        RMSE值
    """
    return torch.sqrt(F.mse_loss(pred, target)).item()


def psnr_metric(pred, target):
    """
    计算PSNR指标
    
    Args:
        pred: 预测图像 (numpy array)
        target: 目标图像 (numpy array)
        
    Returns:
        PSNR值
    """
    return psnr(target, pred, data_range=pred.max() - pred.min())


def ssim_metric(pred, target):
    """
    计算SSIM指标
    
    Args:
        pred: 预测图像 (numpy array)
        target: 目标图像 (numpy array)
        
    Returns:
        SSIM值
    """
    # 确保输入是numpy数组并且是正确的形状
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
        
    # 处理批次维度
    if pred.ndim == 4:  # (B, C, H, W)
        pred = pred.squeeze(1) if pred.shape[1] == 1 else pred
    if target.ndim == 4:  # (B, C, H, W)
        target = target.squeeze(1) if target.shape[1] == 1 else target
        
    # 处理单张图像情况
    if pred.ndim == 2 and target.ndim == 2:
        return ssim(target, pred, data_range=pred.max() - pred.min())
    
    # 处理批次图像情况
    if pred.ndim == 3 and target.ndim == 3:
        total_ssim = 0
        for i in range(pred.shape[0]):
            total_ssim += ssim(target[i], pred[i], data_range=pred[i].max() - pred[i].min())
        return total_ssim / pred.shape[0]
    
    return 0.0


def calculate_all_metrics(pred, target):
    """
    计算所有指标
    
    Args:
        pred: 预测图像
        target: 目标图像
        
    Returns:
        包含所有指标的字典
    """
    # 转换为numpy数组用于某些指标计算
    pred_np = pred.detach().cpu().numpy() if torch.is_tensor(pred) else pred
    target_np = target.detach().cpu().numpy() if torch.is_tensor(target) else target
    
    metrics = {
        'mse': mse_metric(pred, target),
        'rmse': rmse_metric(pred, target),
    }
    
    try:
        metrics['psnr'] = psnr_metric(pred_np, target_np)
    except:
        metrics['psnr'] = 0.0
        
    try:
        metrics['ssim'] = ssim_metric(pred_np, target_np)
    except:
        metrics['ssim'] = 0.0
    
    return metrics