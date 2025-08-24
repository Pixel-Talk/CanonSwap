import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 1) 基础Block: 2D卷积 + InstanceNorm + ReLU
# ---------------------------
class SameBlock2d(nn.Module):
    """
    不改变分辨率的卷积层：Conv2d + InstanceNorm2d + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.inorm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.inorm(x)
        x = self.relu(x)
        return x


class DownBlock2d(nn.Module):
    """
    下采样层：Conv2d(stride=2) + InstanceNorm2d + ReLU，分辨率减半
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=2,
                              padding=padding)
        self.inorm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.inorm(x)
        x = self.relu(x)
        return x

class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm3d(in_features, affine=True)
        self.norm2 = nn.InstanceNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = out + x
        return out

# ---------------------------
# 2) ID Encoder: 更深更宽
#    输出 shape: B x 512 x 16 x 16
# ---------------------------
class IdentityEncoder(nn.Module):
    def __init__(self, 
                 image_channel=3,
                 base_channels=32,
                 max_channels =128,
                 num_down_blocks=4):
        super(IdentityEncoder, self).__init__()
        
        # 1) 初始SameBlock
        self.initial = SameBlock2d(image_channel, base_channels)

        # 2) 下采样
        down_blocks = []
        in_ch = base_channels
        for i in range(num_down_blocks):
            out_ch = min(max_channels, in_ch * 2)
            down_blocks.append(DownBlock2d(in_ch, out_ch))
            in_ch = out_ch

        self.down_blocks = nn.ModuleList(down_blocks)

        # 3) 最终 1x1 卷积，通道固定为 max_channels
        self.final_conv = nn.Conv2d(in_ch, max_channels, kernel_size=1, stride=1)

    def forward(self, x):
        """
        x shape: B x 3 x 256 x 256
        output shape: B x 512 x 16 x 16
        """
        x = self.initial(x)  # -> B x 64 x 256 x 256
        for block in self.down_blocks:  
            x = block(x)      # 最终 -> B x 128 x 16 x 16
        x = self.final_conv(x)  # 保持 B x 128 x 16 x 16
        return x


# ---------------------------
# 3) Attribute Encoder: 更小，下采样较少
#    输出 shape: B x 256 x 64 x 64
# ---------------------------
class AttributeEncoder(nn.Module):
    def __init__(self,
                 image_channel=3,
                 base_channels=32,
                 max_channels=128,
                 num_down_blocks=2):
        super(AttributeEncoder, self).__init__()

        # 1) 初始 SameBlock
        self.initial = SameBlock2d(image_channel, base_channels)

        # 2) 下采样
        down_blocks = []
        in_ch = base_channels
        for i in range(num_down_blocks):
            out_ch = min(max_channels, in_ch * 2)
            down_blocks.append(DownBlock2d(in_ch, out_ch))
            in_ch = out_ch

        self.down_blocks = nn.ModuleList(down_blocks)

        # 3) 最终卷积，得到输出通道=64
        self.final_conv = nn.Conv2d(in_ch, max_channels, kernel_size=1, stride=1)

    def forward(self, x):
        """
        x shape: B x 3 x 256 x 256
        output shape: B x 256 x 64 x 64
        """
        x = self.initial(x)  # -> B x 32 x 256 x 256
        for block in self.down_blocks: 
            x = block(x)      # -> B x 128 x 64 x 64 (after 2 downblocks)
        x = self.final_conv(x)  # -> B x 256 x 64 x 64
        return x


# ---------------------------
# 4) Decoder:
#    - 包含: 上采样(对ID特征), concat, 若干卷积, reshape
#    - 最终输出: B x 32 x 16 x 64 x 64 (即通道=512, reshape为(32,16))
# ---------------------------
class Decoder(nn.Module):
    def __init__(self,
                 id_in_channels=128,      # ID Encoder输出: 512
                 attr_in_channels=128,    # Attribute Encoder输出: 256
                 out_channels=512,        # Decoder最终输出通道
                 reshape_channel=32,
                 reshape_depth=16,
                 num_resblocks=4):
        super(Decoder, self).__init__()

        # 先定义上采样方法 (这里简单用最近邻插值，你也可以换成转置卷积等)
        # self.upsample_id = nn.Upsample(scale_factor=4, mode='nearest') 
        self.upsample_id = nn.ConvTranspose2d(
            in_channels=id_in_channels,
            out_channels=id_in_channels,  # 保持通道数不变
            kernel_size=4,            # 卷积核大小
            stride=4,                 # 步长，scale_factor=4
            padding=0,                # 无填充
            output_padding=0          # 无额外填充
        )
        # scale_factor=4: 16 -> 64

        # 拼接后的通道: 128 + 128 =256
        in_channels = id_in_channels + attr_in_channels

        # 几个卷积层，用 InstanceNorm 提升容量
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.inorm1 = nn.InstanceNorm2d(128, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.inorm2 = nn.InstanceNorm2d(256, affine=True)
        self.relu2 = nn.ReLU(inplace=True)

        # 最终再做一个 conv, 输出 out_channels=512
        self.conv3 = nn.Conv2d(256, out_channels, kernel_size=3, padding=1)
        self.inorm3 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))
    def forward(self, id_feat, attr_feat):
        """
        id_feat:   B x 512 x 16 x 16
        attr_feat: B x 256 x 64 x 64

        目标:
         1) upsample id_feat -> B x 512 x 64 x 64
         2) concat -> B x 768 x 64 x 64
         3) conv -> B x 512 x 64 x 64
         4) reshape -> B x 32 x 16 x 64 x 64
        """
        # 1) 上采样 ID特征: 16 -> 64
        id_up = self.upsample_id(id_feat)  # B x 512 x 64 x 64

        # 2) 拼接
        x = torch.cat([id_up, attr_feat], dim=1)  # B x 768 x 64 x 64

        # 3) 经过若干卷积层
        x = self.relu1(self.inorm1(self.conv1(x)))  # -> B x 512 x 64 x 64
        x = self.relu2(self.inorm2(self.conv2(x)))  # -> B x 512 x 64 x 64
        x = self.relu3(self.inorm3(self.conv3(x)))  # -> B x 512 x 64 x 64

        # 4) reshape -> 3D
        b, c, h, w = x.shape  # c=512, h=64, w=64
        assert c == self.reshape_channel * self.reshape_depth, (
            f"通道数{c}与reshape要求{self.reshape_channel}*{self.reshape_depth}不符!"
        )
        out_3d = x.view(b, self.reshape_channel, self.reshape_depth, h, w)  
        # => B x 32 x 16 x 64 x 64
        out_3d = self.resblocks_3d(out_3d)  # -> B x 32 x 16 x 64 x 64
        return out_3d


# ---------------------------
# 5) 整合: DualEncoderDecoder
# ---------------------------
class DualEncoderDecoder(nn.Module):
    def __init__(self):
        super(DualEncoderDecoder, self).__init__()

        self.id_encoder = IdentityEncoder(
            image_channel=3,
            base_channels=64,   # 初始通道
            max_channels=512,
            num_down_blocks=4   # 共 4 次下采样 => 16x16
        )
        self.attr_encoder = AttributeEncoder(
            image_channel=3,
            base_channels=32,
            max_channels=256,
            num_down_blocks=2   # 2 次下采样 => 64x64
        )

        self.decoder = Decoder(
            id_in_channels=512,
            attr_in_channels=256,
            out_channels=512,   # 最终要 reshape 成 32*16=512
            reshape_channel=32,
            reshape_depth=16
        )

    def forward(self, id_img, attr_img):
        """
        id_img:   B x 3 x 256 x 256
        attr_img: B x 3 x 256 x 256
        最终输出: B x 32 x 16 x 64 x 64
        """
        id_feat = self.id_encoder(id_img)         # => B x 512 x 16 x 16
        attr_feat = self.attr_encoder(attr_img)   # => B x 256 x 64 x 64

        out_3d = self.decoder(id_feat, attr_feat) # => B x 32 x 16 x 64 x 64
        return out_3d