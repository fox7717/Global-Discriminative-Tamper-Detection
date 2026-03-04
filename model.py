import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 关键词1: 双残差网络 (Dual Residual Network) ---
class DualResBlock(nn.Module):
    """包含两个并行卷积路径的双残差块"""
    def __init__(self, in_channels, out_channels):
        super(DualResBlock, self).__init__()
        # 路径A
        self.pathA = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # 路径B (残差捷径)
        self.pathB = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.pathB(x)
        out = self.pathA(x)
        out += residual # 残差连接
        return self.relu(out)

# --- 关键词2: 全局相关性 & 非局部注意力 (Global Correlation) ---
class GlobalCorrelationBlock(nn.Module):
    """计算特征图的全局空间相关性"""
    def __init__(self, in_channels, reduction=8):
        super(GlobalCorrelationBlock, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        self.query = nn.Conv2d(in_channels, in_channels//reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels//reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1)) # 可学习的权重

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # 计算Query和Key的点积，建立全局相关性矩阵
        proj_query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1) # B, HW, C'
        proj_key = self.key(x).view(batch_size, -1, H*W) # B, C', HW
        energy = torch.bmm(proj_query, proj_key) # B, HW, HW (全局相关性矩阵)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, H*W) # B, C, HW
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, HW
        out = out.view(batch_size, C, H, W)
        
        # 加上原始特征 (非局部残差连接)
        out = self.gamma * out + x 
        return out

# --- 关键词3: 区分性增强 (Discriminative Enhancement) ---
class DiscriminativeEnhancement(nn.Module):
    """通过通道注意力增强关键特征"""
    def __init__(self, channel, reduction=16):
        super(DiscriminativeEnhancement, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 增强区分性特征
        return x * y.expand_as(x)

# --- 关键词4: 边界监督 (Boundary Supervision) ---
class BoundarySupervisionNet(nn.Module):
    """主网络：包含区域检测和边界检测双分支"""
    def __init__(self, n_classes=1):
        super(BoundarySupervisionNet, self).__init__()
        
        # 编码器 (使用双残差块)
        self.enc1 = DualResBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DualResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # 核心增强模块
        self.global_corr = GlobalCorrelationBlock(128) # 全局相关性
        self.discrim_enhance = DiscriminativeEnhancement(128) # 区分性增强
        
        # 解码器
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec = DualResBlock(64, 64)
        
        # 双分支输出 (边界监督)
        self.region_head = nn.Conv2d(64, n_classes, kernel_size=1) # 区域预测
        self.boundary_head = nn.Conv2d(64, n_classes, kernel_size=1) # 边界预测

    def forward(self, x):
        # 编码
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        
        # 核心处理
        x2 = self.global_corr(x2)
        x2 = self.discrim_enhance(x2)
        
        # 解码
        x = self.up(x2)
        x = self.dec(x)
        
        # 边界监督双输出
        region_map = torch.sigmoid(self.region_head(x))
        boundary_map = torch.sigmoid(self.boundary_head(x))
        
        return region_map, boundary_map
