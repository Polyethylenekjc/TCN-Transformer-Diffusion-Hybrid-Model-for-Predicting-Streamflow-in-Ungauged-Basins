import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
from .spatial_encoder import SpatialEncoder
from .temporal_encoder import TemporalTCN
from .transformer_encoder import TransformerEncoder
from .conditional_unet import ConditionalUNet
import time


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


class StationLossCalculator:
    """
    站点损失计算器，用于计算基于观测站点数据的径流约束损失
    """
    def __init__(self, stations_csv, resolution=1.0, loss_type='l1', lat_range=(0.0, 256.0), lon_range=(0.0, 256.0)):
        """
        初始化站点损失计算器
        
        Args:
            stations_csv: 包含站点信息的CSV文件路径，应包含列：lat, lon, runoff
            resolution: 输出图像的地理分辨率（度），默认为1.0度
            loss_type: 损失函数类型 ('l1' 或 'mse')
            lat_range: 纬度范围 (min, max)
            lon_range: 经度范围 (min, max)
        """
        self.stations_df = pd.read_csv(stations_csv)
        self.resolution = resolution
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("loss_type must be 'l1' or 'mse'")
    
    def _get_pixel_coords(self, lat, lon, image_height, image_width):
        """
        将经纬度转换为图像像素坐标
        
        Args:
            lat: 纬度
            lon: 经度
            image_height: 图像高度
            image_width: 图像宽度
            
        Returns:
            (row, col): 像素坐标，如果超出范围则返回(None, None)
        """
        # 检查是否在图像地理范围内
        if not (self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max):
            return None, None

        # 将经纬度线性映射到图像像素坐标
        # 注意：纬度(lat)对应行(row)，经度(lon)对应列(col)
        # 并且确保结果在 [0, size) 范围内
        row = int((lat - self.lat_min) / (self.lat_max - self.lat_min) * image_height)
        col = int((lon - self.lon_min) / (self.lon_max - self.lon_min) * image_width)

        # 由于浮点数精度问题，可能会出现等于size的情况，需要修正
        row = min(row, image_height - 1)
        col = min(col, image_width - 1)

        return row, col
    
    def calculate_loss(self, pred_images, target_runoff_values):
        """
        计算站点损失
        
        Args:
            pred_images: 预测图像 [B, 1, H, W]
            target_runoff_values: 目标径流值 [N_stations] 或 [N_stations, B]
            
        Returns:
            站点损失值
        """
        batch_size, _, height, width = pred_images.shape
        total_loss = 0.0
        valid_station_count = 0
        
        # 检查target_runoff_values的维度
        if target_runoff_values.dim() == 1:
            # 如果是1维，说明是[N_stations]格式
            target_runoff_values = target_runoff_values.unsqueeze(1)  # 变为[N_stations, 1]
        
        for idx, (_, station) in enumerate(self.stations_df.iterrows()):
            lat, lon = station['lat'], station['lon']
            
            # 获取像素坐标
            row, col = self._get_pixel_coords(lat, lon, height, width)
            
            # 如果站点在图像范围内
            if row is not None and col is not None:
                # 提取预测值（在站点位置的像素值）
                pred_value = pred_images[:, :, row, col]  # [B, 1]
                
                # 获取目标值，确保索引不会越界
                if target_runoff_values.shape[0] <= idx:
                    # 如果索引超出范围，跳过该站点
                    continue
                    
                if target_runoff_values.shape[1] == 1:
                    # 如果目标值只有一列，则扩展到batch_size
                    target_values = target_runoff_values[idx].unsqueeze(0).expand(batch_size, 1)  # [B, 1]
                else:
                    # 如果目标值有多列，则直接取对应的batch_size个值，但要确保不会索引越界
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


class ForecastingModel(nn.Module):
    """
    完整的预测模型，整合所有组件：
    Spatial CNN → Temporal TCN → Global Transformer → Conditional Diffusion
    """
    def __init__(self, 
                 input_channels,
                 input_height=32,
                 input_width=32,
                 output_height=128,
                 output_width=128,
                 d_model=64,
                 tcn_dilations=[1, 2],
                 transformer_num_heads=4,
                 transformer_num_layers=2,
                 diffusion_time_steps=100,
                 stations_csv=None,
                 lambda_station=0.1,
                 lat_range=(0.0, 256.0),
                 lon_range=(0.0, 256.0),
                 resolution=1.0):
        """
        Args:
            input_channels: 输入多通道图像的通道数
            input_height: 输入图像高度
            input_width: 输入图像宽度
            output_height: 输出图像高度
            output_width: 输出图像宽度
            d_model: 特征维度
            tcn_dilations: TCN膨胀系数
            transformer_num_heads: Transformer注意力头数
            transformer_num_layers: Transformer层数
            diffusion_time_steps: 扩散过程的时间步数
            stations_csv: 站点数据CSV文件路径
            lambda_station: 站点损失权重
            lat_range: 纬度范围 (min, max)
            lon_range: 经度范围 (min, max)
            resolution: 地理分辨率（度）
        """
        super(ForecastingModel, self).__init__()
        
        # 存储尺寸信息
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        
        # 组件初始化
        self.spatial_encoder = SpatialEncoder(input_channels, d_model=d_model, downsample_steps=2)
        self.temporal_encoder = TemporalTCN(d_model, dilations=tcn_dilations)
        # 使用temporal_encoder的实际输出维度来初始化transformer_encoder
        transformer_d_model = self.temporal_encoder.output_dim
        self.transformer_encoder = TransformerEncoder(
            d_model=transformer_d_model, 
            num_heads=transformer_num_heads, 
            num_layers=transformer_num_layers,
            hidden_dim=2*transformer_d_model,
            max_seq_length=256
        )
        self.diffusion_decoder = ConditionalUNet(
            in_channels=1, 
            out_channels=1, 
            d_model=transformer_d_model  # 使用transformer的实际输出维度
        )
        
        self.d_model = d_model
        self.transformer_d_model = transformer_d_model
        self.diffusion_time_steps = diffusion_time_steps
        
        # 站点损失相关
        self.stations_csv = stations_csv
        self.lambda_station = lambda_station
        if stations_csv:
            self.station_loss_calculator = StationLossCalculator(
                stations_csv, 
                resolution=resolution,
                lat_range=lat_range,
                lon_range=lon_range
            )
        else:
            self.station_loss_calculator = None
        
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
        temporal_features = self.temporal_encoder(spatial_features)  # (B, temporal_d_model, H', W')
        
        # Transformer全局上下文建模
        cond_features = self.transformer_encoder(temporal_features)  # (B, temporal_d_model, H', W')
        
        return cond_features
    
    def forward_diffusion(self, x_seq, y_gt, timestep=None, target_runoff_values=None):
        """
        扩散训练前向过程
        
        Args:
            x_seq: 输入序列，形状为 (B, T, C, H_in, W_in)
            y_gt: 真实标签，形状为 (B, 1, H_out, W_out)
            timestep: 时间步，如果为None则随机采样
            target_runoff_values: 目标径流值 [N_stations]，如果提供则计算站点损失
            
        Returns:
            预测噪声和真实噪声，如果提供target_runoff_values还会返回站点损失
        """
        # 获取条件编码
        cond = self.forward_frame_prediction(x_seq)  # (B, transformer_d_model, H', W')
        
        # 确保y_gt在正确的设备上
        y_gt = y_gt.to(cond.device)
        
        # 在加噪前计算站点损失（避免OOM问题）
        station_loss = None
        if target_runoff_values is not None and self.station_loss_calculator is not None:
            station_loss = self.station_loss_calculator.calculate_loss(y_gt, target_runoff_values)
        
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
        
        if station_loss is not None:
            return predicted_noise, noise, station_loss
        else:
            return predicted_noise, noise

    def sample(self, x_seq, target_runoff_values=None):
        """
        从噪声中采样生成最终图像
        
        Args:
            x_seq: 输入序列，形状为 (B, T, C, H_in, W_in)
            target_runoff_values: 目标径流值 [N_stations]，如果提供则计算站点损失
            
        Returns:
            生成的图像，形状为 (B, 1, H_out, W_out)
            如果提供target_runoff_values，还会返回站点损失
        """
        # 获取条件编码
        cond = self.forward_frame_prediction(x_seq)  # (B, transformer_d_model, H', W')
        
        # 初始化
        batch_size = x_seq.shape[0]
        # 使用配置的输出尺寸
        image_shape = (batch_size, 1, self.output_height, self.output_width)
        x = torch.randn(image_shape, device=cond.device)
        
        # 逐步去噪
        for t in reversed(range(0, 50)):  # 默认使用50步采样
            # 计算当前时间步
            timestep = torch.full((batch_size,), t, device=cond.device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.diffusion_decoder(x, timestep, cond)
            
            # 更新图像
            x = self._denoise_step(x, predicted_noise, timestep, image_shape)
            
        # 确保输出尺寸正确
        if x.shape[2] != self.output_height or x.shape[3] != self.output_width:
            x = F.interpolate(x, size=(self.output_height, self.output_width), mode='bilinear', align_corners=True)
            
        # 如果提供了目标径流值，则计算站点损失
        if target_runoff_values is not None and self.station_loss_calculator is not None:
            station_loss = self.station_loss_calculator.calculate_loss(x, target_runoff_values)
            return x, station_loss
            
        return x

    def sample_with_station_loss(self, x_seq, target_runoff_values=None):
        """
        从噪声中采样生成最终图像，并可选择计算站点损失（优化内存使用）
        
        Args:
            x_seq: 输入序列，形状为 (B, T, C, H_in, W_in)
            target_runoff_values: 目标径流值 [N_stations]，如果提供则计算站点损失
            
        Returns:
            生成的图像，形状为 (B, 1, H_out, W_out)
            如果提供target_runoff_values，还会返回站点损失
        """
        # 获取条件编码
        cond = self.forward_frame_prediction(x_seq)  # (B, transformer_d_model, H', W')
        
        # 初始化
        batch_size = x_seq.shape[0]
        # 使用配置的输出尺寸
        image_shape = (batch_size, 1, self.output_height, self.output_width)
        x = torch.randn(image_shape, device=cond.device)
        
        # 逐步去噪
        for t in reversed(range(0, 50)):  # 默认使用50步采样
            # 计算当前时间步
            timestep = torch.full((batch_size,), t, device=cond.device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.diffusion_decoder(x, timestep, cond)
            
            # 更新图像
            x = self._denoise_step(x, predicted_noise, timestep, image_shape)
            
        # 确保输出尺寸正确
        if x.shape[2] != self.output_height or x.shape[3] != self.output_width:
            x = F.interpolate(x, size=(self.output_height, self.output_width), mode='bilinear', align_corners=True)
            
        # 如果提供了目标径流值，则计算站点损失
        if target_runoff_values is not None and self.station_loss_calculator is not None:
            # 使用torch.no_grad()上下文管理器，避免保留计算图
            with torch.no_grad():
                station_loss = self.station_loss_calculator.calculate_loss(x, target_runoff_values)
            return x, station_loss
            
        return x

    def compute_station_loss_only(self, x_seq, generated_image, target_runoff_values):
        """
        仅计算站点损失，不保留完整计算图以减少内存消耗
        
        Args:
            x_seq: 输入序列，形状为 (B, T, C, H_in, W_in)
            generated_image: 生成的图像，形状为 (B, 1, H_out, W_out)
            target_runoff_values: 目标径流值 [N_stations]
            
        Returns:
            站点损失值
        """
        if self.station_loss_calculator is not None:
            # 使用torch.no_grad()上下文管理器，避免保留计算图
            with torch.no_grad():
                station_loss = self.station_loss_calculator.calculate_loss(generated_image, target_runoff_values)
            return station_loss
        else:
            return torch.tensor(0.0, device=generated_image.device)

    def _denoise_step(self, x, predicted_noise, timestep, target_shape):
        """
        单步去噪过程 - 使用标准DDPM公式
        """
        # 调整噪声张量大小
        if predicted_noise.shape != x.shape:
            predicted_noise = F.interpolate(predicted_noise, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # 获取当前时间步的系数
        t = timestep[0].item()
        beta_t = self._betas[t]
        alpha_t = self._alphas[t]
        alpha_cumprod_t = self._alphas_cumprod[t]
        
        # 计算后验方差
        posterior_variance = torch.zeros_like(beta_t) if t == 0 else \
            beta_t * (1. - self._alphas_cumprod[t-1]) / (1. - alpha_cumprod_t)
        
        # 计算均值
        coeff1 = 1. / torch.sqrt(alpha_t)
        coeff2 = (1. - alpha_t) / torch.sqrt(1. - alpha_cumprod_t)
        mean = coeff1 * (x - coeff2 * predicted_noise)
        
        # 添加噪声（最后一层除外）
        result = mean if t == 0 else mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
            
        # 调整输出尺寸
        if result.shape[2] != target_shape[2] or result.shape[3] != target_shape[3]:
            result = F.interpolate(result, size=(target_shape[2], target_shape[3]), mode='bilinear', align_corners=True)
            
        return result

    def _extract_coefficient(self, coefficient_array, t, x_shape):
        """
        从系数数组中提取指定时间步的系数
        """
        batch_size = t.shape[0]
        # 确保系数数组在正确的设备上
        coefficient_array = coefficient_array.to(t.device)
        out = coefficient_array.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def _precompute_coefficients(self):
        """
        预计算扩散过程中的系数
        """
        # 这些参数在实际应用中应该根据具体的扩散调度策略设置
        betas = torch.linspace(1e-4, 0.02, self.diffusion_time_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        self.register_buffer('_betas', betas)
        self.register_buffer('_alphas', alphas)
        self.register_buffer('_alphas_cumprod', alphas_cumprod)
        self.register_buffer('_sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('_sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def forward(self, x_seq, y_gt=None, mode='train', target_runoff_values=None):
        """
        前向传播
        
        Args:
            x_seq: 输入序列，形状为 (B, T, C, H_in, W_in)
            y_gt: 真实标签，形状为 (B, 1, H_out, W_out)
            mode: 运行模式 ('train' 或 'sample')
            target_runoff_values: 目标径流值 [N_stations]，如果提供则计算站点损失
            
        Returns:
            训练模式: (predicted_noise, true_noise) 或 (predicted_noise, true_noise, station_loss)
            采样模式: 生成的图像 (B, 1, H_out, W_out)
        """
        if mode == 'train':
            assert y_gt is not None, "y_gt must be provided in train mode"
            return self.forward_diffusion(x_seq, y_gt, target_runoff_values=target_runoff_values)
        elif mode == 'sample':
            if target_runoff_values is not None:
                return self.sample_with_station_loss(x_seq, target_runoff_values)
            else:
                return self.sample(x_seq)
        else:
            raise ValueError(f"Unknown mode: {mode}")
