#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整的测试流程，包括数据生成、模型训练、评估和可视化
"""

import torch
import torch.nn as nn
import sys
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from models.full_model import ForecastingModel, initialize_model
from utils.losses import CombinedLoss, TotalLoss
from utils.metrics import calculate_all_metrics
from utils.logger import create_logger


def load_config(config_path):
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 将嵌套的配置展平为单层字典以保持向后兼容性
    flat_config = {}
    for section, values in config.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
            
    return flat_config


class MockDatasetWithStations(torch.utils.data.Dataset):
    """
    模拟带有站点数据的数据集
    """
    def __init__(self, size=100, sequence_length=15, input_size=(128, 128), output_size=(256, 256), 
                 input_channels=10, stations_csv="data/stations.csv"):
        self.size = size
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        self.input_channels = input_channels
        
        # 加载站点数据
        self.stations_df = pd.read_csv(stations_csv)
        self.num_stations = len(self.stations_df)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 生成输入序列 [T, C, H, W]
        sequence = torch.randn(self.sequence_length, self.input_channels, *self.input_size)
        
        # 生成目标图像 [1, H_out, W_out]
        target = torch.randn(1, *self.output_size)
        
        # 生成站点径流值 [N_stations] 或 [N_stations, T_target]
        # 这里我们简化为 [N_stations]，表示单个时间步的观测值
        station_runoffs = torch.randn(self.num_stations)
        
        return sequence, target, station_runoffs


def create_data_loader(dataset, batch_size=4, shuffle=True):
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        
    Returns:
        DataLoader对象
    """
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def visualize_samples(x_seq, y_gt, generated=None, batch_idx=0, save_path="./samples"):
    """
    可视化样本
    
    Args:
        x_seq: 输入序列 (B, T, C, H, W)
        y_gt: 真实标签 (B, 1, H_out, W_out)
        generated: 生成的图像 (B, 1, H_out, W_out)
        batch_idx: 批次索引
        save_path: 保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 可视化输入序列的几个帧
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Batch {batch_idx} - Input Sequence and Ground Truth')
    
    # 显示输入序列的前5帧
    for i in range(min(5, x_seq.shape[1])):
        # 选择第一个通道进行可视化
        frame = x_seq[0, i, 0].cpu().numpy()
        axes[0, i].imshow(frame, cmap='viridis')
        axes[0, i].set_title(f'Frame {i+1}')
        axes[0, i].axis('off')
    
    # 显示真实标签
    gt_image = y_gt[0, 0].cpu().numpy()
    axes[1, 0].imshow(gt_image, cmap='viridis')
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    # 如果有生成结果，显示生成图像
    if generated is not None:
        gen_image = generated[0, 0].cpu().numpy()
        axes[1, 1].imshow(gen_image, cmap='viridis')
        axes[1, 1].set_title('Generated')
        axes[1, 1].axis('off')
        
        # 显示差异
        diff = np.abs(gt_image - gen_image)
        axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('Difference')
        axes[1, 2].axis('off')
    
    # 隐藏多余的子图
    for i in range(3, 5):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"sample_batch_{batch_idx}.png"))
    plt.close()


def train_model(model, train_loader, val_loader, config, logger):
    """
    训练模型
    
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置
        logger: 日志记录器
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    
    # 混合精度训练设置
    use_mixed_precision = config.get('use_mixed_precision', False)
    if use_mixed_precision:
        # 使用新API解决FutureWarning
        if hasattr(torch.amp, 'GradScaler'):
            scaler = torch.amp.GradScaler('cuda')
        else:
            # 回退到旧API以防新API不可用
            scaler = torch.cuda.amp.GradScaler()
    
    # 损失函数
    if config.get('use_station_loss', False):
        criterion = TotalLoss(
            stations_csv=config.get('csv_file', 'data/stations.csv'),
            resolution=config.get('resolution', 1.0),
            lambda_recon=config.get('lambda_recon', 1.0),
            lambda_station=config.get('lambda_station', 0.1),
            recon_loss_type=config.get('recon_loss_type', 'l1'),
            station_loss_type=config.get('station_loss_type', 'l1')
        )
    else:
        criterion = CombinedLoss(
            lambda_diff=config.get('lambda_diff', 1.0),
            lambda_pix=config.get('lambda_pix', 0.1),
            lambda_perc=config.get('lambda_perc', 0.01)
        )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', 1e-4))
    
    # 训练循环
    num_epochs = config.get('epochs', 3)
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (x_seq, y_gt, station_runoffs) in enumerate(train_loader):
            x_seq = x_seq.to(device)
            y_gt = y_gt.to(device)
            station_runoffs = station_runoffs.to(device)
            
            optimizer.zero_grad()
            
            if config.get('use_station_loss', False):
                # 使用站点损失的训练方式
                # 混合精度训练
                if use_mixed_precision:
                    # 解决FutureWarning: 使用新的API
                    with (torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') 
                          else torch.cuda.amp.autocast()):
                        # 直接在加噪前计算站点损失，避免内存消耗
                        result = model(x_seq, y_gt, mode='train', target_runoff_values=station_runoffs)
                        
                        if len(result) == 3:
                            predicted_noise, true_noise, station_loss = result
                        else:
                            predicted_noise, true_noise = result
                            station_loss = torch.tensor(0.0, device=device)
                        
                        # 计算扩散损失
                        diffusion_loss = criterion.diffusion_loss(predicted_noise, true_noise) if hasattr(criterion, 'diffusion_loss') else nn.MSELoss()(predicted_noise, true_noise)
                        
                        # 计算总损失
                        lambda_station = config.get('lambda_station', 0.1)
                        total_loss = diffusion_loss + lambda_station * station_loss
                        
                        # 仅记录重建损失用于日志输出
                        recon_loss = diffusion_loss
                else:
                    # 直接在加噪前计算站点损失，避免内存消耗
                    result = model(x_seq, y_gt, mode='train', target_runoff_values=station_runoffs)
                    
                    if len(result) == 3:
                        predicted_noise, true_noise, station_loss = result
                    else:
                        predicted_noise, true_noise = result
                        station_loss = torch.tensor(0.0, device=device)
                    
                    # 计算扩散损失
                    diffusion_loss = criterion.diffusion_loss(predicted_noise, true_noise) if hasattr(criterion, 'diffusion_loss') else nn.MSELoss()(predicted_noise, true_noise)
                    
                    # 计算总损失
                    lambda_station = config.get('lambda_station', 0.1)
                    total_loss = diffusion_loss + lambda_station * station_loss
                    
                    # 仅记录重建损失用于日志输出
                    recon_loss = diffusion_loss
                
                if use_mixed_precision:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()
                
                train_loss += total_loss.item()
                train_batches += 1
                
                # 每隔一定批次记录一次
                if batch_idx % 5 == 0:
                    logger.info(f"  Batch {batch_idx}, Total Loss: {total_loss.item():.4f}, "
                                f"Recon Loss: {recon_loss.item():.4f}, Station Loss: {station_loss.item():.4f}")
            else:
                # 原始训练方式
                # 混合精度训练
                if use_mixed_precision:
                    # 解决FutureWarning: 使用新的API
                    with (torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') 
                          else torch.cuda.amp.autocast()):
                        predicted_noise, true_noise = model(x_seq, y_gt, mode='train')
                        
                        # 计算损失
                        loss = criterion(predicted_noise, true_noise)
                else:
                    predicted_noise, true_noise = model(x_seq, y_gt, mode='train')
                    
                    # 计算损失
                    loss = criterion(predicted_noise, true_noise)
                
                # 反向传播
                if use_mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # 每隔一定批次记录一次
                if batch_idx % 5 == 0:
                    logger.info(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # 移除可视化部分以减少内存消耗
            # if batch_idx < 3:
            #     model.eval()
            #     with torch.no_grad():
            #         # 混合精度推理
            #         if use_mixed_precision:
            #             with torch.cuda.amp.autocast():
            #                 if config.get('use_station_loss', False):
            #                     generated = generated  # 已经生成过了
            #                 else:
            #                     generated = model(x_seq, mode='sample')
            #         else:
            #             if config.get('use_station_loss', False):
            #                 generated = generated  # 已经生成过了
            #             else:
            #                 generated = model(x_seq, mode='sample')
            #         visualize_samples(x_seq, y_gt, generated, batch_idx)
            #     model.train()
        
        avg_train_loss = train_loss / train_batches
        logger.info(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        total_metrics = {'mse': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0}
        
        with torch.no_grad():
            for batch_idx, (x_seq, y_gt, station_runoffs) in enumerate(val_loader):
                x_seq = x_seq.to(device)
                y_gt = y_gt.to(device)
                station_runoffs = station_runoffs.to(device)
                
                if config.get('use_station_loss', False):
                    # 使用站点损失的验证方式
                    # 混合精度推理
                    if use_mixed_precision:
                        # 解决FutureWarning: 使用新的API
                        with (torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') 
                              else torch.cuda.amp.autocast()):
                            # 生成图像并计算站点损失
                            result = model(x_seq, target_runoff_values=station_runoffs, mode='sample')
                            if isinstance(result, tuple):
                                generated, station_loss = result
                            else:
                                generated = result
                                station_loss = torch.tensor(0.0, device=device)
                            
                            # 计算重建损失
                            recon_loss = nn.L1Loss()(generated, y_gt)
                            
                            # 计算总损失
                            lambda_station = config.get('lambda_station', 0.1)
                            lambda_recon = config.get('lambda_recon', 1.0)
                            total_loss = lambda_recon * recon_loss + lambda_station * station_loss
                    else:
                        # 生成图像并计算站点损失
                        result = model(x_seq, target_runoff_values=station_runoffs, mode='sample')
                        if isinstance(result, tuple):
                            generated, station_loss = result
                        else:
                            generated = result
                            station_loss = torch.tensor(0.0)
                        
                        # 计算重建损失
                        recon_loss = nn.L1Loss()(generated, y_gt)
                        
                        # 计算总损失
                        lambda_station = config.get('lambda_station', 0.1)
                        lambda_recon = config.get('lambda_recon', 1.0)
                        total_loss = lambda_recon * recon_loss + lambda_station * station_loss
                    
                    val_loss += total_loss.item()
                else:
                    # 原始验证方式
                    # 混合精度推理
                    if use_mixed_precision:
                        # 解决FutureWarning: 使用新的API
                        with (torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') 
                              else torch.cuda.amp.autocast()):
                            predicted_noise, true_noise = model(x_seq, y_gt, mode='train')
                            
                            # 计算损失
                            loss = criterion(predicted_noise, true_noise)
                    else:
                        predicted_noise, true_noise = model(x_seq, y_gt, mode='train')
                        
                        # 计算损失
                        loss = criterion(predicted_noise, true_noise)
                    val_loss += loss.item()
                
                val_batches += 1
                
                # 计算评估指标
                if not config.get('use_station_loss', False):
                    # 只有在不使用站点损失时才需要单独生成
                    if use_mixed_precision:
                        # 解决FutureWarning: 使用新的API
                        with (torch.amp.autocast('cuda') if hasattr(torch.amp, 'autocast') 
                              else torch.cuda.amp.autocast()):
                            generated = model(x_seq, mode='sample')
                    else:
                        generated = model(x_seq, mode='sample')
                # 如果使用站点损失，generated已经生成
                
                batch_metrics = calculate_all_metrics(generated, y_gt)
                
                for key in total_metrics:
                    total_metrics[key] += batch_metrics.get(key, 0)
                
                # 移除验证集可视化以减少内存消耗
                # if epoch == num_epochs - 1 and batch_idx < 2:  # 最后一轮可视化
                #     visualize_samples(x_seq, y_gt, generated, f"val_{batch_idx}")
        
        avg_val_loss = val_loss / val_batches
        for key in total_metrics:
            total_metrics[key] /= val_batches
            
        logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        logger.info("Validation Metrics:")
        for key, value in total_metrics.items():
            logger.info(f"  {key.upper()}: {value:.4f}")


def main():
    """
    主函数
    """
    # 创建日志记录器
    logger = create_logger('full_test', max_log_files=10)
    logger.info("Starting full test pipeline...")
    
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 加载低内存配置
    config = load_config('configs/low_memory.yaml')
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 创建数据集
    logger.info("Creating datasets...")
    input_size = (config['input_height'], config['input_width'])
    output_size = (config['output_height'], config['output_width'])
    stations_csv_path = config.get('csv_file', 'data/stations.csv')  # 使用 get 方法提供默认值

    train_dataset = MockDatasetWithStations(
        size=10,  # 减少数据集大小
        sequence_length=config['sequence_length'],
        input_size=input_size,
        output_size=output_size,
        input_channels=config['input_channels'],
        stations_csv=stations_csv_path
    )
    val_dataset = MockDatasetWithStations(
        size=4,   # 减少数据集大小
        sequence_length=config['sequence_length'],
        input_size=input_size,
        output_size=output_size,
        input_channels=config['input_channels'],
        stations_csv=stations_csv_path
    )
    
    train_loader = create_data_loader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # 创建模型
    logger.info("Creating model...")
    model = ForecastingModel(
        input_channels=config['input_channels'],
        input_height=config['input_height'],
        input_width=config['input_width'],
        output_height=config['output_height'],
        output_width=config['output_width'],
        d_model=config['d_model'],
        tcn_dilations=config['tcn_dilations'],
        transformer_num_heads=config['transformer_num_heads'],
        transformer_num_layers=config['transformer_num_layers'],
        diffusion_time_steps=config['diffusion_time_steps'],
        stations_csv=config.get('csv_file', None),
        lambda_station=config.get('lambda_station', 0.1)
    )
    
    model = initialize_model(model)
    logger.log_model_info(model)
    
    # 训练模型
    logger.info("Starting training...")
    train_model(model, train_loader, val_loader, config, logger)
    
    # 保存模型
    model_path = "./checkpoints/test_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Full test pipeline completed successfully!")


if __name__ == "__main__":
    main()