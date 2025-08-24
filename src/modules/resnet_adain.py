import torch.nn as nn
import torch
from .util import SameBlock2d, DownBlock2d, ResBlock3d
# 使用内置 InstanceNorm3d
InstanceNorm3d = nn.InstanceNorm3d

class ApplyStyle3D(nn.Module):
    """
    与原 ApplyStyle 类似，但针对 3D Feature (N, C, D, H, W)
    """
    def __init__(self, latent_size, channels):
        super(ApplyStyle3D, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        # latent => [batch_size, latent_size]
        style = self.linear(latent)   # => [batch_size, channels*2]
        # 3D 特征: x.shape = [N, C, D, H, W]
        # 因此需要 reshape 为 [N, 2, C, 1, 1, 1]
        shape = [-1, 2, x.size(1), 1, 1, 1]
        style = style.view(shape)

        # style[:, 0] 对应每个通道的 scale
        # style[:, 1] 对应每个通道的 bias
        # 公式： x_out = x * (scale + 1) + bias
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x

class ResnetBlock_Adain3D(nn.Module):
    def __init__(self, dim = 32, latent_size = 512, padding_type='reflect', activation=nn.ReLU(True)):
        """
        dim:           通道数
        latent_size:   风格向量长度
        padding_type:  [reflect | replicate | zero]
        activation:    激活函数 (默认为 nn.ReLU(True))
        """
        super(ResnetBlock_Adain3D, self).__init__()

        # conv1
        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] is not implemented')

        # 卷积 + 3D InstanceNorm
        conv1 += [
            nn.Conv3d(dim, dim, kernel_size=3, padding=p),
            nn.InstanceNorm3d(dim, affine=True)  # 根据需要可设置 affine=True
        ]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle3D(latent_size, dim)
        self.act1 = activation

        # conv2
        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] is not implemented')

        conv2 += [
            nn.Conv3d(dim, dim, kernel_size=3, padding=p),
            nn.InstanceNorm3d(dim, affine=False)
        ]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle3D(latent_size, dim)

    def forward(self, x, dlatents_in_slice):
        """
        x:                  [N, C, D, H, W]
        dlatents_in_slice:  [N, latent_size] (对每个 batch 有一个风格向量)
        """
        # 1) 第一次卷积 + AdaIN 风格
        y = self.conv1(x)              # => [N, C, D, H, W]
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)

        # 2) 第二次卷积 + AdaIN 风格
        y = self.conv2(y)             # => [N, C, D, H, W]
        y = self.style2(y, dlatents_in_slice)

        # 3) 残差连接
        out = x + y
        return out

class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x

class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding = p), nn.InstanceNorm2d(512,affine=True)]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), nn.InstanceNorm2d(512,affine=True)]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)


    def forward(self, x, dlatents_in_slice, dlatents_in_slice2=None):
        if dlatents_in_slice2 is None:
            dlatents_in_slice2 = dlatents_in_slice
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice2)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out

class transfer_model_2id(nn.Module):
    def __init__(self, latent_dim = 512, n_blocks=7, padding_type = 'reflect'):
        assert (n_blocks >= 0)
        super(transfer_model_2id, self).__init__()
        # BN_in = []
        # activation = nn.ReLU(True)
        # for i in range(3):
        #     BN_in += [
        #         ResnetBlock_Adain3D(32, latent_size=latent_dim, padding_type=padding_type, activation=activation)
        #     ]
        # self.BottleNeck_3din = nn.Sequential(*BN_in)
        BN = []
        activation = nn.ReLU(True)

        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(512, latent_size=latent_dim, padding_type=padding_type, activation=activation)
            ]
        self.BottleNeck = nn.Sequential(*BN)

        # BN_out = []
        # activation = nn.ReLU(True)
        # for i in range(3):
        #     BN_out += [
        #         ResnetBlock_Adain3D(32, latent_size=latent_dim, padding_type=padding_type, activation=activation)
        #     ]
        # self.BottleNeck_3dout = nn.Sequential(*BN_out)
        self.resblocks_3d = nn.Sequential()
        for i in range(6):
            self.resblocks_3d.add_module(
                '3dr' + str(i),
                ResBlock3d(32, kernel_size=3, padding=1)
            )
        self.fc = nn.Linear(256*7*7, 512)

    def forward(self, x, dlatents, dlatents2):
        # for i in range(len(self.BottleNeck_3din)):
        #     x = self.BottleNeck_3din[i](x, dlatents)

        bs, c, d, h, w = x.shape
        x = x.view(bs, c*d, h, w)

        dlatents2 = self.fc(dlatents2)
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents, dlatents2)
        x = x.view(bs, c, d, h, w)
        x = self.resblocks_3d(x)
        # for i in range(len(self.BottleNeck_3dout)):
        #     x = self.BottleNeck_3dout[i](x, dlatents)
        return x


class transfer_model(nn.Module):
    def __init__(self, latent_dim = 512, n_blocks=7, padding_type = 'reflect'):
        assert (n_blocks >= 0)
        super(transfer_model, self).__init__()
        # BN_in = []
        # activation = nn.ReLU(True)
        # for i in range(3):
        #     BN_in += [
        #         ResnetBlock_Adain3D(32, latent_size=latent_dim, padding_type=padding_type, activation=activation)
        #     ]
        # self.BottleNeck_3din = nn.Sequential(*BN_in)
        BN = []
        activation = nn.ReLU(True)

        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(512, latent_size=latent_dim, padding_type=padding_type, activation=activation)
            ]
        self.BottleNeck = nn.Sequential(*BN)

        # BN_out = []
        # activation = nn.ReLU(True)
        # for i in range(3):
        #     BN_out += [
        #         ResnetBlock_Adain3D(32, latent_size=latent_dim, padding_type=padding_type, activation=activation)
        #     ]
        # self.BottleNeck_3dout = nn.Sequential(*BN_out)
        self.resblocks_3d = nn.Sequential()
        for i in range(6):
            self.resblocks_3d.add_module(
                '3dr' + str(i),
                ResBlock3d(32, kernel_size=3, padding=1)
            )

    def forward(self, x, dlatents):
        # for i in range(len(self.BottleNeck_3din)):
        #     x = self.BottleNeck_3din[i](x, dlatents)

        bs, c, d, h, w = x.shape
        x = x.view(bs, c*d, h, w)

        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)
        x = x.view(bs, c, d, h, w)
        x = self.resblocks_3d(x)
        # for i in range(len(self.BottleNeck_3dout)):
        #     x = self.BottleNeck_3dout[i](x, dlatents)
        return x


if __name__ == "__main__":
    model = transfer_model()
    total_params = sum(p.numel() for p in model.parameters())
    print("total parameters:", total_params)
