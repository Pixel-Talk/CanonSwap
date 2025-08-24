import torch
import torch.nn as nn
import torch.nn.functional as F
# 输入特征 (Bx32x16x64x64)
#         |
#         |-- Identity Encoder --> Bx128x2x8x8
#         |
#         |-- Attribute Encoder --> Bx512x8x32x32
#         |
#         |-- Decoder (融合 Identity 和 Attribute) --> Bx32x16x64x64

class ResidualBlock3D_DSep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D_DSep, self).__init__()
        self.conv1 = DepthwiseSeparableConv3D(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv3D(out_channels, out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv3D, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.SyncBatchNorm(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, d, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # 全局平均池化
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock3D_SE(nn.Module):
    """带有Squeeze-and-Excitation的3D残差块，使用InstanceNorm3d"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D_SE, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out
        
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.SyncBatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.SyncBatchNorm(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, d, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # 全局平均池化
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock3D_SE(nn.Module):
    """带有Squeeze-and-Excitation的3D残差块，使用InstanceNorm3d"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D_SE, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out
    
class IdentityEncoder(nn.Module):
    def __init__(self, input_channels=32, latent_channels=128):
        super(IdentityEncoder, self).__init__()
        # 第1层
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=3, stride=1, padding=1)  # Bx64x16x64x64
        self.bn1 = nn.SyncBatchNorm(64)
        self.relu1 = nn.ReLU()
        self.res_block1 = ResidualBlock3D_DSep(64, 64)

        # 第2层
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)  # Bx128x8x32x32
        self.bn2 = nn.SyncBatchNorm(128)
        self.relu2 = nn.ReLU()
        self.res_block2 = ResidualBlock3D_DSep(128, 128)

        # 第3层
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)  # Bx256x4x16x16
        self.bn3 = nn.SyncBatchNorm(256)
        self.relu3 = nn.ReLU()
        self.res_block3 = ResidualBlock3D_DSep(256, 256)

        # 第4层
        self.conv4 = nn.Conv3d(256, latent_channels, kernel_size=3, stride=2, padding=1)  # Bxlatent_channelsx2x8x8
        self.bn4 = nn.SyncBatchNorm(latent_channels)
        self.relu4 = nn.ReLU()
        self.res_block4 = ResidualBlock3D_DSep(latent_channels, latent_channels)

    def forward(self, x):
        # 第1层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.res_block1(x)

        # 第2层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.res_block2(x)

        # 第3层
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.res_block3(x)

        # 第4层
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.res_block4(x)

        return x  # 输出形状: Bxlatent_channelsx2x8x8

    

class AttributeEncoder(nn.Module):
    def __init__(self, input_channels=32, output_channels=512):
        super(AttributeEncoder, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv3d(input_channels, 128, kernel_size=3, stride=2, padding=1)  # Bx128x8x32x32
        self.relu1 = nn.ReLU()
        self.bn1 = nn.SyncBatchNorm(128)
        
        # 第二层卷积
        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)  # Bx256x8x32x32
        self.relu2 = nn.ReLU()
        self.bn2 = nn.SyncBatchNorm(256)
        
        # 第三层卷积
        self.conv3 = nn.Conv3d(256, output_channels, kernel_size=3, stride=1, padding=1)  # Bx512x8x32x32
        self.relu3 = nn.ReLU()
        self.bn3 = nn.SyncBatchNorm(output_channels)

    def forward(self, x):

        x = self.conv1(x)

        
        x = self.relu1(x)

        x = self.bn1(x)
        
        # 按照同样的方式检查每一层
        x = self.conv2(x)

        x = self.relu2(x)

        x = self.bn2(x)

        
        x = self.conv3(x)

        x = self.relu3(x)
        x = self.bn3(x)
        return x

    
class Decoder(nn.Module):
    def __init__(self, latent_dim_identity=128, latent_dim_attribute=512, output_channels=32):
        super(Decoder, self).__init__()
        # 上采样身份特征: Bx128x2x8x8 -> Bx128x8x32x32
        self.upsample_identity = nn.ConvTranspose3d(
            in_channels=latent_dim_identity,
            out_channels=latent_dim_identity,
            kernel_size=(4, 4, 4),
            stride=(4, 4, 4),
            padding=0
        )
        self.bn_identity = nn.SyncBatchNorm(latent_dim_identity)
        self.relu_identity = nn.ReLU()
        
        # 特征融合后处理
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=latent_dim_identity + latent_dim_attribute,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.SyncBatchNorm(256),
            nn.ReLU(),
            ResidualBlock3D(256, 256)
        )
        
        # 上采样到16x64x64
        self.upsample = nn.ConvTranspose3d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.bn_up = nn.SyncBatchNorm(128)
        self.relu_up = nn.ReLU()
        self.res_up = ResidualBlock3D(128, 128)
        
        # 最终卷积层
        self.final_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=output_channels,
                kernel_size=3,
                padding=1
            ),
            nn.SyncBatchNorm(output_channels),
            nn.ReLU()  # 确保没有使用原地操作
        )
    
    def forward(self, identity, attribute):
        """
        identity: Bx128x2x8x8
        attribute: Bx512x8x32x32
        """
        # 上采样身份特征
        identity_upsampled = self.relu_identity(self.bn_identity(self.upsample_identity(identity)))  # Bx128x8x32x32
        
        # 特征融合
        fused = torch.cat((identity_upsampled, attribute), dim=1)  # Bx640x8x32x32
        fused = self.fuse_conv(fused)  # Bx256x8x32x32
        
        # 上采样到16x64x64
        upsampled = self.relu_up(self.bn_up(self.upsample(fused)))  # Bx128x16x64x64
        upsampled = self.res_up(upsampled)  # Bx128x16x64x64
        
        # 最终卷积
        out = self.final_conv(upsampled)  # Bx32x16x64x64
        return out

class EnhancedIdentityEncoder(nn.Module):
    def __init__(self, input_channels=32, latent_channels=128):
        super(EnhancedIdentityEncoder, self).__init__()
        # 第1层
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=3, stride=1, padding=1)  # Bx64x16x64x64
        self.in1 = nn.InstanceNorm3d(64, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.res_block1 = ResidualBlock3D_SE(64, 64)
        
        # 第2层
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)  # Bx128x8x32x32
        self.in2 = nn.InstanceNorm3d(128, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.res_block2 = ResidualBlock3D_SE(128, 128)
        self.res_block3 = ResidualBlock3D_SE(128, 128)
        
        # 第3层
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)  # Bx256x4x16x16
        self.in3 = nn.InstanceNorm3d(256, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.res_block4 = ResidualBlock3D_SE(256, 256)
        self.res_block5 = ResidualBlock3D_SE(256, 256)
        
        # 第4层
        self.conv4 = nn.Conv3d(256, latent_channels, kernel_size=3, stride=2, padding=1)  # Bx128x2x8x8
        self.in4 = nn.InstanceNorm3d(latent_channels, affine=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.res_block6 = ResidualBlock3D_SE(latent_channels, latent_channels)
        self.res_block7 = ResidualBlock3D_SE(latent_channels, latent_channels)
    
    def forward(self, x):
        # 第1层
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)
        x = self.res_block1(x)
        
        # 第2层
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu2(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # 第3层
        x = self.conv3(x)
        x = self.in3(x)
        x = self.relu3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        
        # 第4层
        x = self.conv4(x)
        x = self.in4(x)
        x = self.relu4(x)
        x = self.res_block6(x)
        x = self.res_block7(x)
        
        return x  # 输出形状: Bxlatent_channelsx2x8x8
    
# 定义完整模型
class FaceFeatureDecouplingModel(nn.Module):
    def __init__(self, input_channels=32, latent_dim_identity=128, latent_dim_attribute=512, output_channels=32):
        super(FaceFeatureDecouplingModel, self).__init__()
        self.identity_encoder = IdentityEncoder(input_channels, latent_dim_identity)
        self.attribute_encoder = AttributeEncoder(input_channels, latent_dim_attribute)
        self.decoder = Decoder(latent_dim_identity, latent_dim_attribute, output_channels)
    
    def forward(self, x):
        identity = self.identity_encoder(x)    # Bx128x2x8x8
        attribute = self.attribute_encoder(x)  # Bx512x8x32x32
        decoded_feature = self.decoder(identity, attribute)  # Bx32x16x64x64
        return decoded_feature, identity, attribute


class EnhancedAttributeEncoder(nn.Module):
    def __init__(self, input_channels=32, output_channels=512):
        super(EnhancedAttributeEncoder, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv3d(input_channels, 128, kernel_size=3, stride=2, padding=1)  # Bx128x8x32x32
        self.in1 = nn.InstanceNorm3d(128, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.res_block1 = ResidualBlock3D_SE(128, 128)
        
        # 第二层卷积
        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)  # Bx256x4x16x16
        self.in2 = nn.InstanceNorm3d(256, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.res_block2 = ResidualBlock3D_SE(256, 256)
        
        # 第三层卷积
        self.conv3 = nn.Conv3d(256, output_channels, kernel_size=3, stride=2, padding=1)  # Bx512x2x8x8
        self.in3 = nn.InstanceNorm3d(output_channels, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.res_block3 = ResidualBlock3D_SE(output_channels, output_channels)
        self.res_block4 = ResidualBlock3D_SE(output_channels, output_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)
        x = self.res_block1(x)
        
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu2(x)
        x = self.res_block2(x)
        
        x = self.conv3(x)
        x = self.in3(x)
        x = self.relu3(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        return x  # 输出形状: Bx512x2x8x8

class EnhancedDecoder(nn.Module):
    def __init__(self, latent_dim_identity=128, latent_dim_attribute=512, output_channels=32):
        super(EnhancedDecoder, self).__init__()
        # 上采样身份特征: Bx128x2x8x8 -> Bx128x4x16x16
        self.upsample_identity1 = nn.ConvTranspose3d(
            in_channels=latent_dim_identity,
            out_channels=latent_dim_identity,
            kernel_size=(2, 4, 4),
            stride=(2, 4, 4),
            padding=0
        )
        self.in_identity1 = nn.InstanceNorm3d(latent_dim_identity, affine=True)
        self.relu_identity1 = nn.ReLU(inplace=True)
        self.res_block_identity1 = ResidualBlock3D_SE(latent_dim_identity, latent_dim_identity)
        
        # 再次上采样到 Bx128x8x32x32
        self.upsample_identity2 = nn.ConvTranspose3d(
            in_channels=latent_dim_identity,
            out_channels=latent_dim_identity,
            kernel_size=(2, 4, 4),
            stride=(2, 4, 4),
            padding=0
        )
        self.in_identity2 = nn.InstanceNorm3d(latent_dim_identity, affine=True)
        self.relu_identity2 = nn.ReLU(inplace=True)
        self.res_block_identity2 = ResidualBlock3D_SE(latent_dim_identity, latent_dim_identity)
        
        # 特征融合后处理
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=latent_dim_identity + latent_dim_attribute,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.InstanceNorm3d(512),
            nn.ReLU(inplace=True),
            ResidualBlock3D_SE(512, 512),
            ResidualBlock3D_SE(512, 256)
        )
        
        # 上采样到 Bx128x16x64x64
        self.upsample1 = nn.ConvTranspose3d(
            in_channels=256,
            out_channels=128,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            padding=0
        )
        self.in_up1 = nn.InstanceNorm3d(128, affine=True)
        self.relu_up1 = nn.ReLU(inplace=True)
        self.res_up1 = ResidualBlock3D_SE(128, 128)
        
        # 上采样到 Bx64x32x128x128
        self.upsample2 = nn.ConvTranspose3d(
            in_channels=128,
            out_channels=64,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            padding=0
        )
        self.in_up2 = nn.InstanceNorm3d(64, affine=True)
        self.relu_up2 = nn.ReLU(inplace=True)
        self.res_up2 = ResidualBlock3D_SE(64, 64)
        
        # 最终卷积层
        self.final_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=output_channels,
                kernel_size=3,
                padding=1
            ),
            nn.InstanceNorm3d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, identity, attribute):
        """
        identity: Bx128x2x8x8
        attribute: Bx512x2x8x8
        """
        # 上采样身份特征到 Bx128x4x16x16
        identity_upsampled = self.relu_identity1(self.in_identity1(self.upsample_identity1(identity)))  # Bx128x4x16x16
        identity_upsampled = self.res_block_identity1(identity_upsampled)
        
        # 再次上采样到 Bx128x8x32x32
        identity_upsampled = self.relu_identity2(self.in_identity2(self.upsample_identity2(identity_upsampled)))  # Bx128x8x32x32
        identity_upsampled = self.res_block_identity2(identity_upsampled)
        
        # 特征融合
        fused = torch.cat((identity_upsampled, attribute), dim=1)  # Bx640x8x32x32
        fused = self.fuse_conv(fused)  # Bx256x8x32x32
        
        # 上采样到 Bx128x16x64x64
        upsampled = self.relu_up1(self.in_up1(self.upsample1(fused)))  # Bx128x16x64x64
        upsampled = self.res_up1(upsampled)  # Bx128x16x64x64
        
        # 上采样到 Bx64x32x128x128
        upsampled = self.relu_up2(self.in_up2(self.upsample2(upsampled)))  # Bx64x32x128x128
        upsampled = self.res_up2(upsampled)  # Bx64x32x128x128
        
        # 最终卷积
        out = self.final_conv(upsampled)  # Bx32x32x128x128
        
        # 调整到 Bx32x16x64x64
        out = F.interpolate(out, size=(16, 64, 64), mode="trilinear", align_corners=False)  # Bx32x16x64x64
        
        return out
    
class SimplifiedIdentityEncoder(nn.Module):
    def __init__(self, input_channels=32, latent_channels=128):
        super(SimplifiedIdentityEncoder, self).__init__()
        
        # 第1层卷积：下采样并增加通道数
        self.conv1 = nn.Conv3d(input_channels, 128, kernel_size=3, stride=2, padding=1)  # Bx128x8x32x32
        self.in1 = nn.InstanceNorm3d(128, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.res_block1 = ResidualBlock3D_SE(128, 128)
        
        # 第2层卷积：下采样并增加通道数
        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)  # Bx256x4x16x16
        self.in2 = nn.InstanceNorm3d(256, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.res_block2 = ResidualBlock3D_SE(256, 256)
        # self.res_block3 = ResidualBlock3D_SE(256, 256)
        
        # 第3层卷积：下采样并增加通道数
        self.conv3 = nn.Conv3d(256, latent_channels, kernel_size=3, stride=2, padding=1)  # Bx128x2x8x8
        self.in3 = nn.InstanceNorm3d(latent_channels, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.res_block4 = ResidualBlock3D_SE(latent_channels, latent_channels)
        # self.res_block5 = ResidualBlock3D_SE(latent_channels, latent_channels)
    
    def forward(self, x):
        # 第1层
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)
        x = self.res_block1(x)
        
        # 第2层
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu2(x)
        x = self.res_block2(x)
        # x = self.res_block3(x)
        
        # 第3层
        x = self.conv3(x)
        x = self.in3(x)
        x = self.relu3(x)
        x = self.res_block4(x)
        # x = self.res_block5(x)
        
        return x  # 输出形状: Bx128x2x8x8
    

class SimplifiedAttributeEncoder(nn.Module):
    def __init__(self, input_channels=32, output_channels=128):
        super(SimplifiedAttributeEncoder, self).__init__()
        
        # 第1层卷积：下采样并增加通道数
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=3, stride=2, padding=1)  # Bx64x8x32x32
        self.in1 = nn.InstanceNorm3d(64, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.res_block1 = ResidualBlock3D_SE(64, 64)
        
        # 第2层卷积：保持空间维度，增加通道数
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)  # Bx128x8x32x32
        self.in2 = nn.InstanceNorm3d(128, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.res_block2 = ResidualBlock3D_SE(128, 128)
        
        # 第3层卷积：进一步增加通道数
        self.conv3 = nn.Conv3d(128, output_channels, kernel_size=3, stride=1, padding=1)  # Bx512x8x32x32
        self.in3 = nn.InstanceNorm3d(output_channels, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.res_block3 = ResidualBlock3D_SE(output_channels, output_channels)
    
    def forward(self, x):
        # 第1层
        x = self.conv1(x)     # Bx64x8x32x32
        x = self.in1(x)
        x = self.relu1(x)
        x = self.res_block1(x)
        
        # 第2层
        x = self.conv2(x)     # Bx128x8x32x32
        x = self.in2(x)
        x = self.relu2(x)
        x = self.res_block2(x)
        
        # 第3层
        x = self.conv3(x)     # Bx512x8x32x32
        x = self.in3(x)
        x = self.relu3(x)
        x = self.res_block3(x)
        
        return x  # 输出形状: Bx512x8x32x32
    
class SimplifiedDecoder(nn.Module):
    def __init__(self, latent_dim_identity=128, latent_dim_attribute=128, output_channels=32):
        super(SimplifiedDecoder, self).__init__()
        
        # 上采样身份特征: Bx128x2x8x8 -> Bx128x4x16x16
        self.upsample_identity1 = nn.ConvTranspose3d(
            in_channels=latent_dim_identity,
            out_channels=latent_dim_identity,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.in_identity1 = nn.InstanceNorm3d(latent_dim_identity, affine=True)
        self.relu_identity1 = nn.ReLU(inplace=True)
        self.res_block_identity1 = ResidualBlock3D_SE(latent_dim_identity, latent_dim_identity)
        
        # 再次上采样身份特征到 Bx128x8x32x32
        self.upsample_identity2 = nn.ConvTranspose3d(
            in_channels=latent_dim_identity,
            out_channels=latent_dim_identity,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.in_identity2 = nn.InstanceNorm3d(latent_dim_identity, affine=True)
        self.relu_identity2 = nn.ReLU(inplace=True)
        self.res_block_identity2 = ResidualBlock3D_SE(latent_dim_identity, latent_dim_identity)
        
        # 特征融合后处理
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=latent_dim_identity + latent_dim_attribute,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.InstanceNorm3d(256, affine=True),
            nn.ReLU(inplace=True),
            ResidualBlock3D_SE(256, 256)
        )
        
        # 上采样到 Bx256x8x32x32 -> Bx128x16x64x64
        self.upsample1 = nn.ConvTranspose3d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.in_up1 = nn.InstanceNorm3d(128, affine=True)
        self.relu_up1 = nn.ReLU(inplace=True)
        self.res_up1 = ResidualBlock3D_SE(128, 128)
        
        # 上采样到 Bx128x16x64x64 -> Bx64x32x128x128
        self.upsample2 = nn.ConvTranspose3d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.in_up2 = nn.InstanceNorm3d(64, affine=True)
        self.relu_up2 = nn.ReLU(inplace=True)
        self.res_up2 = ResidualBlock3D_SE(64, 64)
        
        # 最终卷积层，调整通道数到输出通道
        self.final_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=64,
                out_channels=output_channels,
                kernel_size=3,
                padding=1
            ),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, identity, attribute):
        """
        identity: Bx128x2x8x8
        attribute: Bx512x8x32x32
        """
        # 上采样身份特征到 Bx128x4x16x16
        identity_upsampled = self.relu_identity1(self.in_identity1(self.upsample_identity1(identity)))  # Bx128x4x16x16
        identity_upsampled = self.res_block_identity1(identity_upsampled)
        
        # 再次上采样身份特征到 Bx128x8x32x32
        identity_upsampled = self.relu_identity2(self.in_identity2(self.upsample_identity2(identity_upsampled)))  # Bx128x8x32x32
        identity_upsampled = self.res_block_identity2(identity_upsampled)
        
        # 特征融合
        fused = torch.cat((identity_upsampled, attribute), dim=1)  # Bx128+512=640x8x32x32
        fused = self.fuse_conv(fused)  # Bx256x8x32x32
        
        # 上采样到 Bx256x8x32x32 -> Bx128x16x64x64
        upsampled = self.relu_up1(self.in_up1(self.upsample1(fused)))  # Bx128x16x64x64
        upsampled = self.res_up1(upsampled)  # Bx128x16x64x64
        
        # 上采样到 Bx128x16x64x64 -> Bx64x32x128x128
        upsampled = self.relu_up2(self.in_up2(self.upsample2(upsampled)))  # Bx64x32x128x128
        upsampled = self.res_up2(upsampled)  # Bx64x32x128x128
        
        # 最终卷积
        out = self.final_conv(upsampled)  # Bx32x32x128x128
        
        # 如果需要调整到 Bx32x16x64x64，可以通过下采样
        out = F.interpolate(out, size=(16, 64, 64), mode="trilinear", align_corners=False)  # Bx32x16x64x64
        
        return out
    
# 测试 AttributeEncoder
if __name__ == "__main__":
    # 假设输入的形状是 Bx32x16x64x64
    input_tensor = torch.randn(2, 32, 16, 64, 64)  # Batch size: 8
    model = FaceFeatureDecouplingModel()
    output_tensor = model(input_tensor)
    print(output_tensor[0].shape)  # 期待输出 Bx512x8x32x32