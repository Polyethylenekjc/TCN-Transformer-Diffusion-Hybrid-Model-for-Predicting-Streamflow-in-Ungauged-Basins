import torch
import argparse
import sys
import os
import yaml

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from models.full_model import ForecastingModel, initialize_model
from train.trainer import create_trainer


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


def main():
    """
    主程序入口
    """
    parser = argparse.ArgumentParser(description='Forecasting Model Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'sample'],
                        help='运行模式')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点文件路径')
    
    args = parser.parse_args()
    
    # 加载配置文件
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # 确保设备设置正确
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config['device'] = 'cpu'
    
    # 创建模型
    print("Creating model...")
    model = ForecastingModel(
        input_channels=config['input_channels'],
        d_model=config['d_model'],
        tcn_dilations=config['tcn_dilations'],
        transformer_num_heads=config['transformer_num_heads'],
        transformer_num_layers=config['transformer_num_layers'],
        diffusion_time_steps=config['diffusion_time_steps']
    )
    
    model = initialize_model(model)
    print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        print("Starting training mode...")
        # 这里应该加载数据并开始训练
        # trainer = create_trainer(config)
        # trainer.train(config['epochs'])
        print("Training would start here...")
        
    elif args.mode == 'sample':
        print("Starting sampling mode...")
        # 示例采样过程
        model.eval()
        with torch.no_grad():
            # 创建示例输入 - 根据配置调整尺寸
            x_seq = torch.randn(1, config['sequence_length'], config['input_channels'], 
                               config['input_height'], config['input_width'])  # (B, T, C, H, W)
            print(f"Input sequence shape: {x_seq.shape}")
            
            # 采样生成
            generated = model(x_seq, mode='sample')
            print(f"Generated image shape: {generated.shape}")
            
    print("Process completed.")


if __name__ == "__main__":
    main()