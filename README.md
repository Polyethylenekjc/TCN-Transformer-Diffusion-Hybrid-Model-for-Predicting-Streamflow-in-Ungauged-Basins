# 多通道图像序列预测高分辨率单通道图像系统

本项目实现了一个完整的解决方案，用于使用15天的多通道图像序列预测第16天的高分辨率单通道图像。

## 架构设计

系统采用以下架构：

1. **Spatial CNN**: 对每天的多通道图像做同一套轻量CNN编码
2. **Temporal TCN**: 使用TCN学习时间序列变化规律
3. **Global Transformer**: 使用Transformer学习全局上下文依赖
4. **Conditional DDPM/UNet**: 条件扩散模型生成高分辨率图像

## 项目结构

```
src/
├── models/                 # 模型组件
│   ├── spatial_encoder.py   # 空间特征提取器
│   ├── temporal_encoder.py  # 时序编码器
│   ├── transformer_encoder.py # Transformer编码器
│   ├── conditional_unet.py  # 条件扩散模型
│   └── full_model.py        # 完整模型整合
├── data/                   # 数据处理
│   └── preprocessing.py     # 数据预处理和增强
├── utils/                  # 工具函数
│   ├── losses.py           # 损失函数
│   └── metrics.py          # 评估指标
├── train/                  # 训练相关
│   └── trainer.py          # 训练器
└── main.py                 # 主程序入口
```

## 核心组件说明

### 1. Spatial Encoder (CNN)
对每天的多通道图像进行特征提取，输出低维空间特征图。

### 2. Temporal TCN
使用因果膨胀卷积处理时间序列，捕获局部与中期依赖。

### 3. Transformer Encoder
学习跨空间的大尺度依赖和注意力权重。

### 4. Conditional UNet for Diffusion
基于条件的扩散模型生成高分辨率图像。

## 使用方法

### 安装依赖
```bash
pip install torch torchvision tqdm scikit-image
```

### 训练模型
```bash
python src/main.py --mode train
```

### 生成样本
```bash
python src/main.py --mode sample
```

## 配置参数

主要超参数建议：
- SpatialEncoder 输出通道: d_model = 64 或 128
- Transformer 层数: 4，头数: 8
- TCN 层数: 4，膨胀系数: [1,2,4,8]
- 扩散步数: 训练1000，采样50-200步
- 批次大小: 8-32（根据显存调整）
- 学习率: 1e-4 (AdamW)

## 损失函数

1. **扩散损失**: MSE between true noise and predicted noise
2. **像素损失** (可选): L1 损失
3. **感知损失** (可选): VGG感知损失或SSIM损失

总损失: L = L_diff + λ_pix*L_pix + λ_perc*L_perc

## 数据预处理

- 输入: 15张日尺度多通道图像 (B, 15, C_in, H_in, W_in)
- 标签: 第16天高分辨率单通道图像 (B, 1, H_out, W_out)
- 归一化: 每通道分别做z-score或min-max
- 数据增强: 随机裁剪、旋转、翻转等

## 训练流程

1. 从数据加载器获取批次数据
2. SpatialEncoder对每帧编码
3. TCN处理时间序列
4. Transformer编码为条件特征
5. 随机采样时间步，添加噪声
6. UNet预测噪声并计算损失
7. 反向传播更新参数

## 推理流程

1. 给定输入序列获取条件编码
2. 从噪声开始逐步去噪生成最终图像
3. 可使用classifier-free guidance提高稳定性

# TTF - Time Series Forecasting with Transformers and Diffusion

This project implements a time series forecasting model using transformers and diffusion models.

## Project Structure

```
src/
├── data/              # Data preprocessing modules
├── models/            # Model definitions
├── train/             # Training utilities
├── utils/             # Utility functions
└── main.py           # Main entry point
configs/               # Configuration files
```

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- Other dependencies in `requirements.txt`

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

```bash
python -m src.main --mode train --config configs/default.yaml
```

### Sampling

```bash
python -m src.main --mode sample --config configs/default.yaml
```

## Configuration

The project supports configuration files to adjust model parameters. Example configuration files are provided in the `configs/` directory:

- `configs/default.yaml`: Default model configuration
- `configs/small_model.yaml`: Smaller model for testing
- `configs/large_model.yaml`: Larger model for better performance

### Configuration Options

- `model.input_channels`: Number of input channels
- `model.d_model`: Model dimension
- `model.tcn_dilations`: Dilations for TCN layers
- `model.transformer_num_heads`: Number of attention heads in transformer
- `model.transformer_num_layers`: Number of transformer layers
- `model.diffusion_time_steps`: Number of diffusion time steps
- `training.learning_rate`: Learning rate for training
- `training.batch_size`: Training batch size
- `training.epochs`: Number of training epochs
- `hardware.device`: Device to run on ("cuda" or "cpu")
- `data.sequence_length`: Length of input sequence
- `data.input_height`: Height of input images
- `data.input_width`: Width of input images

## Model Architecture

The model consists of four main components:
1. Spatial Encoder (CNN)
2. Temporal Encoder (TCN)
3. Global Context Encoder (Transformer)
4. Diffusion Decoder (Conditional UNet)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
