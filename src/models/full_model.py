import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .spatial_encoder import SpatialEncoder
from .temporal_encoder import TemporalTCN
from .transformer_encoder import TransformerEncoder
from .conditional_unet import ConditionalUNet


def initialize_model(model):
    """
    初始化模型权重
    
    Args:
        model: 需要初始化的模型
        
    Returns:
        初始化后的模型
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return model


class ForecastingModel(nn.Module):
    """
    完整的预测模型，整合所有组件：
    Spatial CNN → Temporal TCN → Global Transformer → Conditional Diffusion
    """
    def __init__(self, 
                 input_channels,
                 d_model=64,  # 修改为64保持一致性
                 tcn_dilations=[1, 2, 4, 8],
                 transformer_num_heads=8,
                 transformer_num_layers=4,
                 diffusion_time_steps=1000):
        """
        Args:
            input_channels: 输入多通道图像的通道数
            d_model: 特征维度
            tcn_dilations: TCN膨胀系数
            transformer_num_heads: Transformer注意力头数
            transformer_num_layers: Transformer层数
            diffusion_time_steps: 扩散过程的时间步数
        """
        super(ForecastingModel, self).__init__()
        
        # 组件初始化
        self.spatial_encoder = SpatialEncoder(input_channels, d_model=d_model)
        self.temporal_encoder = TemporalTCN(d_model, dilations=tcn_dilations)
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model, 
            num_heads=transformer_num_heads, 
            num_layers=transformer_num_layers,
            hidden_dim=4*d_model,  # 修复hidden_dim
            max_seq_length=256  # 根据实际特征图大小调整
        )
        self.diffusion_decoder = ConditionalUNet(
            in_channels=1, 
            out_channels=1, 
            d_model=d_model
        )
        
        self.d_model = d_model
        self.diffusion_time_steps = diffusion_time_steps
        
        # 预计算扩散系数
        self._precompute_coefficients()

    def forward_frame_prediction(self, x_seq):
        """
        前向传播：从输入序列到条件编码
        
        Args:
            x_seq: 输入序列，形状为 (B, T, C, H_in, W_in)
            
        Returns:
            条件编码，形状为 (B, d_model, H', W')
        """
        # 空间编码
        spatial_features = self.spatial_encoder(x_seq)  # (B, T, d_model, H', W')
        
        # 时序编码
        temporal_features = self.temporal_encoder(spatial_features)  # (B, d_model, H', W')
        
        # Transformer全局上下文建模
        cond_features = self.transformer_encoder(temporal_features)  # (B, d_model, H', W')
        
        return cond_features
    
    def forward_diffusion(self, x_seq, y_gt, timestep=None):
        """
        扩散训练前向过程
        
        Args:
            x_seq: 输入序列，形状为 (B, T, C, H_in, W_in)
            y_gt: 真实标签，形状为 (B, 1, H_out, W_out)
            timestep: 时间步，如果为None则随机采样
            
        Returns:
            预测噪声和真实噪声
        """
        # 获取条件编码
        cond = self.forward_frame_prediction(x_seq)  # (B, d_model, H', W')
        
        # 随机采样时间步
        if timestep is None:
            timestep = torch.randint(0, self.diffusion_time_steps, (y_gt.shape[0],), device=y_gt.device).long()
            
        # 添加噪声
        noise = torch.randn_like(y_gt)
        sqrt_alphas_cumprod_t = self._extract_coefficient(self._sqrt_alphas_cumprod, timestep, y_gt.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_coefficient(self._sqrt_one_minus_alphas_cumprod, timestep, y_gt.shape)
        
        # 根据DDPM公式添加噪声
        noisy_image = sqrt_alphas_cumprod_t * y_gt + sqrt_one_minus_alphas_cumprod_t * noise
        
        # 预测噪声
        predicted_noise = self.diffusion_decoder(noisy_image, timestep, cond)
        
        return predicted_noise, noise

    def sample(self, x_seq, num_steps=50):
        """
        从噪声中采样生成最终图像
        
        Args:
            x_seq: 输入序列，形状为 (B, T, C, H_in, W_in)
            num_steps: 采样步数
            
        Returns:
            生成的图像，形状为 (B, 1, H_out, W_out)
        """
        # 获取条件编码
        cond = self.forward_frame_prediction(x_seq)  # (B, d_model, H', W')
        
        # 初始化
        batch_size = x_seq.shape[0]
        # 使用输入尺寸的4倍作为输出尺寸
        image_shape = (batch_size, 1, x_seq.shape[-2]*4, x_seq.shape[-1]*4)  
        x = torch.randn(image_shape, device=x_seq.device)
        
        # 逐步去噪
        for t in reversed(range(0, num_steps)):
            # 计算当前时间步
            timestep = torch.full((batch_size,), t, device=x_seq.device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.diffusion_decoder(x, timestep, cond)
            
            # 更新图像
            x = self._denoise_step(x, predicted_noise, timestep, image_shape)
            
        return x

    def _denoise_step(self, x, predicted_noise, timestep, target_shape):
        """
        单步去噪过程
        这里为了简化实现，采用基础的DDPM采样方法
        """
        # 确保预测噪声与x具有相同的形状
        if predicted_noise.shape != x.shape:
            # 如果形状不匹配，调整predicted_noise的形状
            predicted_noise = F.interpolate(predicted_noise, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # 这是一个简化的采样实现
        # 实际应用中可能需要更复杂的调度算法如DDIM
        return x - 0.1 * predicted_noise  # 简化版本

    def _extract_coefficient(self, coefficient_array, t, x_shape):
        """
        从系数数组中提取指定时间步的系数
        """
        batch_size = t.shape[0]
        out = coefficient_array.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _precompute_coefficients(self):
        """
        预计算扩散过程中的系数
        """
        # 这些参数在实际应用中应该根据具体的扩散调度策略设置
        betas = torch.linspace(1e-4, 0.02, self.diffusion_time_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        self.register_buffer('_sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('_sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def forward(self, x_seq, y_gt=None, mode='train'):
        """
        前向传播主接口
        
        Args:
            x_seq: 输入序列，形状为 (B, T, C, H_in, W_in)
            y_gt: 真实标签，在训练模式下必需
            mode: 运行模式 ('train', 'eval', 'sample')
            
        Returns:
            根据模式返回不同的结果
        """
        if mode == 'train':
            assert y_gt is not None, "Ground truth is required in training mode"
            return self.forward_diffusion(x_seq, y_gt)
        elif mode == 'sample':
            return self.sample(x_seq)
        else:
            return self.forward_frame_prediction(x_seq)