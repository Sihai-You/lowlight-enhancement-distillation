# models.py - 只保留优化模型及其消融变体
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 将DDNet目录添加到Python的sys.path
sys.path.append('/home/user/DDNet')
try:
    from model import Main as TeacherModel
except ImportError:
    print("⚠️ 无法导入教师模型，请检查DDNet路径")
    TeacherModel = None

# ==================== 优化模型的核心组件 ====================
class SimplifiedChannelAttention(nn.Module):
    """简化高效的通道注意力"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.shared_fc(self.avg_pool(x).view(b, c))
        max_out = self.shared_fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out)
        return x * out.view(b, c, 1, 1)


class AdaptiveResidualModule(nn.Module):
    """自适应残差模块"""
    def __init__(self, channels=3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, x, enhanced):
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        fused = alpha * enhanced + beta * x
        return torch.clamp(fused, 0, 1)


class EnhancedDepthwiseSeparableConv(nn.Module):
    """增强的深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class SkipFusionModule(nn.Module):
    """跳跃连接融合模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        
    def forward(self, fused_feat):
        return self.conv(fused_feat)


# ==================== 主模型：优化模型（已移除解码器残差） ====================
class OptimizedLightweightStudent(nn.Module):
    """优化后的轻量级学生模型 - 根据消融实验结果移除了解码器残差"""
    def __init__(self, base_channels: int = 32):
        super().__init__()
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=False),
        )
        
        # 编码器（根据消融实验，移除了残差块）
        self.encoder1 = nn.Sequential(
            EnhancedDepthwiseSeparableConv(base_channels, base_channels*2),
            nn.MaxPool2d(2)
        )
        
        self.encoder2 = nn.Sequential(
            EnhancedDepthwiseSeparableConv(base_channels*2, base_channels*4),
            # 根据消融实验，移除了这里的残差块
            nn.MaxPool2d(2)
        )
        
        # 瓶颈层（根据消融实验，移除了残差块）
        self.bottleneck = nn.Sequential(
            EnhancedDepthwiseSeparableConv(base_channels*4, base_channels*8),
            # 根据消融实验，移除了这里的残差块
        )
        
        # 简化注意力（根据消融实验，保留，有轻微提升）
        self.channel_attention = SimplifiedChannelAttention(base_channels*8)
        
        # 解码器
        self.decoder1 = nn.Sequential(
            EnhancedDepthwiseSeparableConv(base_channels*8, base_channels*4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # 跳跃连接融合（根据消融实验，保留，有显著提升）
        self.fusion1 = SkipFusionModule(base_channels*6, base_channels*4)
        
        self.decoder2_conv = EnhancedDepthwiseSeparableConv(base_channels*4, base_channels*2)
        self.decoder2_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.fusion2 = SkipFusionModule(base_channels*3, base_channels*2)
        
        # 输出层
        self.output = nn.Sequential(
            EnhancedDepthwiseSeparableConv(base_channels*2, base_channels),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 自适应残差模块（根据消融实验，保留，有显著提升）
        self.adaptive_residual = AdaptiveResidualModule()
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始特征
        x0 = self.initial_conv(x)
        
        # 编码路径
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        
        # 瓶颈层
        bottleneck = self.bottleneck(x2)
        bottleneck = self.channel_attention(bottleneck)
        
        # 解码路径
        d1 = self.decoder1(bottleneck)
        d1_cat = torch.cat([d1, x1], dim=1)
        d1_fused = self.fusion1(d1_cat)
        
        d2 = self.decoder2_conv(d1_fused)
        d2 = self.decoder2_upsample(d2)
        d2_cat = torch.cat([d2, x0], dim=1)
        d2_fused = self.fusion2(d2_cat)
        
        # 输出
        output = self.output(d2_fused)
        
        # 自适应残差融合
        enhanced = self.adaptive_residual(x, output)
        
        return enhanced


# ==================== 消融实验模型变体 ====================
# 注意：所有变体都基于移除了解码器残差的主模型

class OptimizedWithoutAttention(OptimizedLightweightStudent):
    """优化模型的无注意力变体"""
    def __init__(self, base_channels: int = 32):
        super().__init__(base_channels)
        # 移除注意力机制
        self.channel_attention = nn.Identity()
        
    def forward(self, x):
        # 重写forward方法，跳过注意力
        x0 = self.initial_conv(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        
        bottleneck = self.bottleneck(x2)
        # 注意：这里不应用注意力
        
        d1 = self.decoder1(bottleneck)
        d1_cat = torch.cat([d1, x1], dim=1)
        d1_fused = self.fusion1(d1_cat)
        
        d2 = self.decoder2_conv(d1_fused)
        d2 = self.decoder2_upsample(d2)
        d2_cat = torch.cat([d2, x0], dim=1)
        d2_fused = self.fusion2(d2_cat)
        
        output = self.output(d2_fused)
        enhanced = self.adaptive_residual(x, output)
        
        return enhanced


class OptimizedWithoutAdaptiveResidual(OptimizedLightweightStudent):
    """优化模型的无自适应残差变体"""
    def __init__(self, base_channels: int = 32):
        super().__init__(base_channels)
        # 移除自适应残差模块
        self.adaptive_residual = nn.Identity()
        
    def forward(self, x):
        x0 = self.initial_conv(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        
        bottleneck = self.bottleneck(x2)
        bottleneck = self.channel_attention(bottleneck)
        
        d1 = self.decoder1(bottleneck)
        d1_cat = torch.cat([d1, x1], dim=1)
        d1_fused = self.fusion1(d1_cat)
        
        d2 = self.decoder2_conv(d1_fused)
        d2 = self.decoder2_upsample(d2)
        d2_cat = torch.cat([d2, x0], dim=1)
        d2_fused = self.fusion2(d2_cat)
        
        output = self.output(d2_fused)
        # 注意：这里不应用自适应残差
        return torch.clamp(output, 0, 1)


class OptimizedWithReducedChannels(OptimizedLightweightStudent):
    """优化模型的通道数减半变体"""
    def __init__(self, base_channels: int = 16):  # 通道数减半
        super().__init__(base_channels)


class OptimizedWithoutSkipConnections(OptimizedLightweightStudent):
    """优化模型的无跳跃连接变体"""
    def __init__(self, base_channels: int = 32):
        super().__init__(base_channels)
        
    def forward(self, x):
        # 重写forward方法，跳过跳跃连接
        x0 = self.initial_conv(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        
        bottleneck = self.bottleneck(x2)
        bottleneck = self.channel_attention(bottleneck)
        
        # 解码路径（无跳跃连接）
        d1 = self.decoder1(bottleneck)
        # 注意：这里不进行跳跃连接
        d1_fused = d1  # 直接使用decoder输出
        
        d2 = self.decoder2_conv(d1_fused)
        d2 = self.decoder2_upsample(d2)
        # 注意：这里不进行跳跃连接
        d2_fused = d2  # 直接使用decoder输出
        
        # 输出
        output = self.output(d2_fused)
        
        # 自适应残差融合
        enhanced = self.adaptive_residual(x, output)
        
        return enhanced


# 注意：不再需要OptimizedWithoutDecoderResidual类，因为主模型已经移除了解码器残差

# ==================== 消融实验模型工厂 ====================
class AblationModelFactory:
    """消融实验模型工厂 - 完整更新版"""
    
    @staticmethod
    def create_model(model_type, base_channels=None):
        """创建消融实验模型
        
        Args:
            model_type: 模型类型
            base_channels: 基础通道数，如果为None则使用默认值
            
        Returns:
            对应的模型实例
        """
        if base_channels is None:
            from config import ModelConfig
            if model_type == "reduced_channels":
                base_channels = ModelConfig.LIGHTWEIGHT_CHANNELS  # 16通道
            else:
                base_channels = ModelConfig.DEFAULT_BASE_CHANNELS  # 32通道
        
        if model_type == "full" or model_type == "optimized":
            return OptimizedLightweightStudent(base_channels=base_channels)
        elif model_type == "without_attention":
            return OptimizedWithoutAttention(base_channels=base_channels)
        elif model_type == "without_adaptive_residual":
            return OptimizedWithoutAdaptiveResidual(base_channels=base_channels)
        elif model_type == "reduced_channels":
            # 确保通道数不小于8
            reduced_channels = max(8, base_channels // 2)
            return OptimizedWithReducedChannels(base_channels=reduced_channels)
        elif model_type == "without_skip":
            return OptimizedWithoutSkipConnections(base_channels=base_channels)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    @staticmethod
    def get_all_ablation_types():
        """获取所有消融实验类型
        
        Returns:
            所有可用的模型类型列表
        """
        return [
            "full",                    # 完整优化模型（已移除解码器残差）
            "without_attention",       # 无注意力机制
            "without_adaptive_residual", # 无自适应残差
            "reduced_channels",        # 通道数减半
            "without_skip",            # 无跳跃连接
        ]
    
    @staticmethod
    def get_model_name(model_type):
        """获取模型的可读名称
        
        Args:
            model_type: 模型类型
            
        Returns:
            模型的可读名称
        """
        names = {
            "full": "完整优化模型",
            "without_attention": "无注意力机制",
            "without_adaptive_residual": "无自适应残差",
            "reduced_channels": "通道数减半",
            "without_skip": "无跳跃连接",
        }
        return names.get(model_type, model_type)
    
    @staticmethod
    def get_model_description(model_type):
        """获取模型的详细描述
        
        Args:
            model_type: 模型类型
            
        Returns:
            模型的详细描述
        """
        descriptions = {
            "full": "基于消融实验优化的完整模型（移除了解码器残差）",
            "without_attention": "移除注意力机制，评估其对性能的影响",
            "without_adaptive_residual": "移除自适应残差融合，评估其对性能的影响",
            "reduced_channels": "通道数减半，评估模型轻量化效果",
            "without_skip": "移除跳跃连接，评估其对信息传递的重要性",
        }
        return descriptions.get(model_type, "未知模型")
    
    @staticmethod
    def get_recommended_model():
        """获取推荐的模型类型
        
        根据消融实验结果，推荐使用通道数减半的轻量模型
        
        Returns:
            推荐的模型类型
        """
        return "reduced_channels"
    
    @staticmethod
    def print_all_models():
        """打印所有可用的模型信息"""
        print("📋 可用的模型类型:")
        print("-" * 50)
        for model_type in AblationModelFactory.get_all_ablation_types():
            name = AblationModelFactory.get_model_name(model_type)
            desc = AblationModelFactory.get_model_description(model_type)
            print(f"  {model_type:<20} - {name}")
            print(f"      {desc}")
            print()