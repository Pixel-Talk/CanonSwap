import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import SameBlock2d, DownBlock2d, ResBlock3d

class ModulatedConv3d(nn.Module):
    """
    参考 StyleGAN2 的 3D 版本示例，用于替代原先的 Conv3d + InstanceNorm3d + AdaIN。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 eps=1e-8):
        super().__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)*3
        self.stride = stride if isinstance(stride, tuple) else (stride,)*3
        self.padding = padding if isinstance(padding, tuple) else (padding,)*3
        self.bias = bias

        # 卷积权重：维度 [out_channels, in_channels, kD, kH, kW]
        # 这里初始化方式可以参考 kaiming_normal 或者 stylegan2 原项目
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, *self.kernel_size) * 0.01)

        # 风格全连接，把 latent 映射到 in_channels
        self.style_fc = nn.Linear(latent_size, in_channels, bias=True)

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias_param = None

    def forward(self, x, latent):
        """
        x: [N, inC, D, H, W]
        latent: [N, latent_size]
        """
        N, _, D, H, W = x.shape
        # 1) 计算对 inC 进行的调制系数 scale => [N, inC]
        style = self.style_fc(latent)  # => [N, inC]
        style = style.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # => [N, inC, 1, 1, 1]

        # 2) 对卷积权重做调制 => w' = w * scale
        # 原始 w.shape = [outC, inC, kD, kH, kW]
        # 调整后 w_mod.shape = [N, outC, inC, kD, kH, kW]
        w = self.weight.unsqueeze(0)  # => [1, outC, inC, kD, kH, kW]
        w_mod = w * style[:, None, :, :, :, :]  # 广播到 [N, outC, inC, kD, kH, kW]

        # 3) Demodulation
        #   每个样本、每个输出通道的范数，用于对 w_mod 做归一化
        #   norm.shape = [N, outC, 1, 1, 1, 1]
        demod = torch.rsqrt((w_mod**2).sum(dim=(2,3,4,5), keepdim=True) + self.eps)
        w_mod = w_mod * demod  # => [N, outC, inC, kD, kH, kW]

        # 4) 组卷积 (group = N)，把 batch 维度展开成 group
        #   x => [1, N*inC, D, H, W]
        #   w_mod => [N*outC, inC, kD, kH, kW] (先把 outC 合并到第一维度)
        x = x.view(1, N*self.in_channels, D, H, W)
        w_mod = w_mod.view(N*self.out_channels, self.in_channels, *self.kernel_size)

        out = F.conv3d(
            x,
            w_mod,
            bias=None,  # 暂时先不加 bias；如果需要则要同样做拆分
            stride=self.stride,
            padding=self.padding,
            groups=N  # 分成 N 组
        )
        # out.shape = [1, N*outC, D, H, W]
        # 还原回 [N, outC, D, H, W]
        out = out.view(N, self.out_channels, D, H, W)

        # 如果需要 bias，则加上
        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1, 1, 1)

        return out


class ModulatedConv2d(nn.Module):
    """
    类似上面 2D 版本，用于替代原先的 Conv2d + InstanceNorm2d + AdaIN。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 eps=1e-8):
        super().__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)*2
        self.stride = stride if isinstance(stride, tuple) else (stride,)*2
        self.padding = padding if isinstance(padding, tuple) else (padding,)*2
        self.bias = bias

        # 卷积权重
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, *self.kernel_size) * 0.01)

        # 风格全连接
        self.style_fc = nn.Linear(latent_size, in_channels, bias=True)

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias_param = None

    def forward(self, x, latent):
        """
        x: [N, inC, H, W]
        latent: [N, latent_size]
        """
        N, _, H, W = x.shape
        # 1) 计算 scale => [N, inC]
        style = self.style_fc(latent)  # => [N, inC]
        style = style.unsqueeze(-1).unsqueeze(-1)  # => [N, inC, 1, 1]

        # 2) 调制权重
        w = self.weight.unsqueeze(0)  # => [1, outC, inC, kH, kW]
        w_mod = w * style[:, None, :, :, :]  # => [N, outC, inC, kH, kW]

        # 3) Demodulation
        demod = torch.rsqrt((w_mod**2).sum(dim=(2,3,4), keepdim=True) + self.eps)
        w_mod = w_mod * demod  # => [N, outC, inC, kH, kW]

        # 4) 组卷积
        x = x.view(1, N*self.in_channels, H, W)
        w_mod = w_mod.view(N*self.out_channels, self.in_channels, *self.kernel_size)

        out = F.conv2d(
            x,
            w_mod,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=N
        )
        out = out.view(N, self.out_channels, out.shape[2], out.shape[3])

        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1, 1)

        return out

class ResnetBlock_StyleGAN2_3D(nn.Module):
    def __init__(self, dim=32, latent_size=512, activation=nn.ReLU(True)):
        super().__init__()
        self.dim = dim
        self.act = activation

        # 两次 ModulatedConv3d
        self.conv1 = ModulatedConv3d(
            in_channels=dim,
            out_channels=dim,
            latent_size=latent_size,
            kernel_size=3,
            padding=1,
            bias=True  # 是否加bias，看你需要
        )
        self.conv2 = ModulatedConv3d(
            in_channels=dim,
            out_channels=dim,
            latent_size=latent_size,
            kernel_size=3,
            padding=1,
            bias=True
        )

    def forward(self, x, dlatents_in_slice):
        """
        x: [N, C, D, H, W]
        dlatents_in_slice: [N, latent_size]
        """
        y = self.conv1(x, dlatents_in_slice)  # => [N, C, D, H, W]
        y = self.act(y)
        y = self.conv2(y, dlatents_in_slice)  # => [N, C, D, H, W]
        return x + y  # ResNet 残差

class ResnetBlock_StyleGAN2_2D(nn.Module):
    def __init__(self, dim=512, latent_size=512, activation=nn.ReLU(True)):
        super().__init__()
        self.dim = dim
        self.act = activation

        self.conv1 = ModulatedConv2d(
            in_channels=dim,
            out_channels=dim,
            latent_size=latent_size,
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.conv2 = ModulatedConv2d(
            in_channels=dim,
            out_channels=dim,
            latent_size=latent_size,
            kernel_size=3,
            padding=1,
            bias=True
        )

    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x, dlatents_in_slice)
        y = self.act(y)
        y = self.conv2(y, dlatents_in_slice)
        return x + y

class transfer_model(nn.Module):
    def __init__(self, latent_dim=512, n_blocks=4, padding_type='reflect'):
        super(transfer_model, self).__init__()
        activation = nn.ReLU(True)

        # 3D in
        BN_in = []
        for i in range(3):
            BN_in += [
                ResnetBlock_StyleGAN2_3D(dim=32, latent_size=latent_dim, activation=activation)
            ]
        self.BottleNeck_3din = nn.Sequential(*BN_in)

        # 2D
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_StyleGAN2_2D(dim=512, latent_size=latent_dim, activation=activation)
            ]
        self.BottleNeck_2d = nn.Sequential(*BN)

        # 3D out
        BN_out = []
        for i in range(3):
            BN_out += [
                ResnetBlock_StyleGAN2_3D(dim=32, latent_size=latent_dim, activation=activation)
            ]
        self.BottleNeck_3dout = nn.Sequential(*BN_out)

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(3):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(32, kernel_size=3, padding=1))

    def forward(self, x, dlatents):
        # x => [N, 32, D, H, W] 假设是这样
        # 1) 3D in
        for i in range(len(self.BottleNeck_3din)):
            x = self.BottleNeck_3din[i](x, dlatents)

        # 2) reshape to 2D => [N, 32*D, H, W]
        bs, c, d, h, w = x.shape
        x = x.view(bs, c*d, h, w)

        # 2D blocks
        for i in range(len(self.BottleNeck_2d)):
            x = self.BottleNeck_2d[i](x, dlatents)

        # reshape back => [N, 32, D, H, W]
        x = x.view(bs, c, d, h, w)

        # 3) 3D out
        for i in range(len(self.BottleNeck_3dout)):
            x = self.BottleNeck_3dout[i](x, dlatents)
        
        x = self.resblocks_3d(x)

        return x

if __name__ == "__main__":
    model = transfer_model()
    total_params = sum(p.numel() for p in model.parameters())
    print("total parameters:", total_params)