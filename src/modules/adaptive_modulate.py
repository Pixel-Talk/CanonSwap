import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import ResBlock3d, ResBlock3D_stage3, ResBlock2d, ResBlock3D_stage3_leak
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1. 自适应共享权重的 2D 卷积
# ------------------------------------------------------------------
class RegionAwareAdaptiveNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-8):
        """
        初始化 Region-Aware Adaptive Normalization 模块。

        Args:
        - in_channels: 输入通道数。
        - eps: 防止除零的小值。
        """
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps

        # 可学习的 gamma 和 beta 分支
        self.gamma_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.beta_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, F, mask):
        """
        应用 Region-Aware Adaptive Normalization。

        Args:
        - F: 输入特征图，形状为 [N, C, H, W]。
        - mask: 掩码，形状为 [N, 1, H, W]，值域在 [0, 1]。

        Returns:
        - F_out: 区域自适应归一化后的特征图。
        """
        N, C, H, W = F.shape

        # 计算 mask 覆盖区域的统计量
        F_masked = F * mask
        mean_masked = F_masked.sum(dim=(2, 3), keepdim=True) / (mask.sum(dim=(2, 3), keepdim=True) + self.eps)
        var_masked = ((F_masked - mean_masked) ** 2 * mask).sum(dim=(2, 3), keepdim=True) / (mask.sum(dim=(2, 3), keepdim=True) + self.eps)
        F_norm_masked = (F_masked - mean_masked) / torch.sqrt(var_masked + self.eps)

        # 计算 mask 外区域的统计量
        complement_mask = 1 - mask
        F_complement = F * complement_mask
        mean_complement = F_complement.sum(dim=(2, 3), keepdim=True) / (complement_mask.sum(dim=(2, 3), keepdim=True) + self.eps)
        var_complement = ((F_complement - mean_complement) ** 2 * complement_mask).sum(dim=(2, 3), keepdim=True) / (complement_mask.sum(dim=(2, 3), keepdim=True) + self.eps)
        F_norm_complement = (F_complement - mean_complement) / torch.sqrt(var_complement + self.eps)

        # 通过 mask 外的区域学习 gamma 和 beta
        gamma = self.gamma_fc(F_complement)  # [N, C, H, W]
        beta = self.beta_fc(F_complement)   # [N, C, H, W]

        # 应用 gamma 和 beta 到归一化后的特征
        F_norm_complement = F_norm_complement * gamma + beta

        # 融合两种区域的特征
        F_out = mask * F_norm_masked + complement_mask * F_norm_complement

        return F_out

class AdaptiveSharedWeightConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_size,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        eps=1e-8,
        use_learned_mask=True,
        use_adaptive_norm=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)*2
        self.stride = stride
        self.padding = padding
        self.bias_flag = bias
        self.eps = eps

        # 单一卷积核权重 (共用)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size) * 0.01
        )

        # 调制分支需要的 style_fc
        hidden_size = in_channels
        self.style_fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(hidden_size, in_channels, bias=True)
        )

        # bias，如果需要的话
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias_param = None

        # 可选：学习一个 mask (也可以外部喂 mask)
        self.use_learned_mask = use_learned_mask
        if use_learned_mask:
            self.mask_conv = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.mask_conv = None

        self.use_adaptive_norm = use_adaptive_norm
        if use_adaptive_norm:
            self.adaptive_norm = RegionAwareAdaptiveNorm(in_channels)
    def forward(self, x, latent, external_mask=None):
        """
        x: [N, inC, H, W]
        latent: [N, latent_size]
        external_mask: [N, 1, H, W] (可选)

        returns: (out, mask)   # mask 维度: [N,1,H,W]
        """
        N, _, H, W = x.shape

        # 1) 标准卷积 out_std
        out_std = F.conv2d(
            x,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding
        )

        # 2) 调制卷积 out_mod
        style = self.style_fc(latent)                # => [N, inC]
        style = style.unsqueeze(-1).unsqueeze(-1)    # => [N, inC, 1, 1]

        w = self.weight.unsqueeze(0)                 # => [1, outC, inC, kH, kW]
        w_mod = w * style[:, None, :, :, :]          # => [N, outC, inC, kH, kW]

        demod = torch.rsqrt((w_mod**2).sum(dim=(2,3,4), keepdim=True) + self.eps)
        w_mod = w_mod * demod                        # => [N, outC, inC, kH, kW]

        x_reshape = x.view(1, N*self.in_channels, H, W)
        w_mod_reshape = w_mod.view(N*self.out_channels, self.in_channels, *self.kernel_size)
        out_mod = F.conv2d(
            x_reshape,
            w_mod_reshape,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=N
        )
        out_mod = out_mod.view(N, self.out_channels, out_mod.shape[2], out_mod.shape[3])

        if self.bias_param is not None:
            out_mod = out_mod + self.bias_param.view(1, -1, 1, 1)

        # 3) 得到 mask
        if external_mask is not None:
            mask = external_mask
        elif self.mask_conv is not None:
            mask = self.mask_conv(x)  # => [N,1,H,W]
        else:
            # 若无 mask_conv，也无 external_mask，就默认全用调制 => mask=1
            mask = torch.ones_like(out_std[:,0:1,:,:])

        # 4) 融合
        # adaptive_intensity = mask * 0.1
        # noise = torch.randn_like(mask, device = mask.device) * adaptive_intensity
        # mask = mask + noise

        out = mask * out_mod + (1 - mask) * out_std
        # out = out_mod

        # 5) 可选：应用自适应归一化
        if self.use_adaptive_norm:
            out = self.adaptive_norm(out, mask)

        return out, mask


# ------------------------------------------------------------------
# 2. 自适应共享权重的 3D 卷积
# ------------------------------------------------------------------
class AdaptiveSharedWeightConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_size,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        eps=1e-8,
        use_learned_mask=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)*3
        self.stride = stride if isinstance(stride, tuple) else (stride,)*3
        self.padding = padding if isinstance(padding, tuple) else (padding,)*3
        self.bias_flag = bias
        self.eps = eps

        # 单一3D卷积核 (共享)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size) * 0.01
        )

        # style_fc: 将 latent => [N, inC]
        hidden_size = in_channels
        self.style_fc = nn.Sequential(
            nn.Linear(latent_size, hidden_size, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(hidden_size, in_channels, bias=True)
        )
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias_param = None

        # 可选：学习一个空间 mask
        self.use_learned_mask = use_learned_mask
        if use_learned_mask:
            self.mask_conv = nn.Sequential(
                nn.Conv3d(in_channels, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.mask_conv = None

    def forward(self, x, latent, external_mask=None):
        """
        x: [N, inC, D, H, W]
        latent: [N, latent_size]
        external_mask: [N, 1, D, H, W], 值在 [0,1]

        returns: (out, mask)   # mask 维度: [N,1,D,H,W]
        """
        N, _, D, H, W = x.shape

        # 1) 标准卷积 => out_std
        out_std = F.conv3d(
            x,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding
        )

        # 2) 调制卷积 => out_mod
        style = self.style_fc(latent)  # => [N, inC]
        style = style.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # => [N, inC, 1, 1, 1]

        w = self.weight.unsqueeze(0)           # => [1, outC, inC, kD, kH, kW]
        w_mod = w * style[:, None, :, :, :, :] # => [N, outC, inC, kD, kH, kW]

        demod = torch.rsqrt((w_mod**2).sum(dim=(2,3,4,5), keepdim=True) + self.eps)
        w_mod = w_mod * demod

        x_reshape = x.view(1, N*self.in_channels, D, H, W)
        w_mod_reshape = w_mod.view(N*self.out_channels, self.in_channels, *self.kernel_size)
        out_mod = F.conv3d(
            x_reshape,
            w_mod_reshape,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=N
        )
        D_out, H_out, W_out = out_mod.shape[2], out_mod.shape[3], out_mod.shape[4]
        out_mod = out_mod.view(N, self.out_channels, D_out, H_out, W_out)

        if self.bias_param is not None:
            out_mod = out_mod + self.bias_param.view(1, -1, 1, 1, 1)

        # 3) 得到 mask
        if external_mask is not None:
            mask = external_mask
        elif self.mask_conv is not None:
            mask = self.mask_conv(x)  # => [N,1,D,H,W]
        else:
            mask = torch.ones_like(out_std[:,0:1,:,:,:])

        # 4) 融合 => out
        out = mask * out_mod + (1 - mask) * out_std

        return out, mask


# ------------------------------------------------------------------
# 3. 2D/3D 自适应残差块，分别返回 (out, mask_of_second_conv)
# ------------------------------------------------------------------
class ResnetBlock_Adaptive2D(nn.Module):
    def __init__(self, dim=512, latent_size=512, activation=nn.ReLU(True), use_adaptive_norm=False):
        super().__init__()
        self.dim = dim
        self.act = activation

        self.conv1 = AdaptiveSharedWeightConv2d(
            in_channels=dim,
            out_channels=dim,
            latent_size=latent_size,
            kernel_size=3,
            padding=1,
            bias=True,
            use_learned_mask=True,
            use_adaptive_norm=use_adaptive_norm
        )
        self.conv2 = AdaptiveSharedWeightConv2d(
            in_channels=dim,
            out_channels=dim,
            latent_size=latent_size,
            kernel_size=3,
            padding=1,
            bias=True,
            use_learned_mask=True,
            use_adaptive_norm=use_adaptive_norm
        )

    def forward(self, x, dlatents_in_slice, dlatents2 = None):
        """
        x: [N, dim, H, W]
        dlatents_in_slice: [N, latent_size]
        returns: (out, mask2) # mask2: [N,1,H,W]
        """
        if dlatents2 is None:
            dlatents2 = dlatents_in_slice
        y, mask1 = self.conv1(x, dlatents2)
        y = self.act(y)
        y, mask2 = self.conv2(y, dlatents_in_slice)
        out = x + y
        return out, (mask1 + mask2)/2


class ResnetBlock_Adaptive3D(nn.Module):
    def __init__(self, dim=32, latent_size=512, activation=nn.ReLU(True)):
        super().__init__()
        self.dim = dim
        self.act = activation

        self.conv1 = AdaptiveSharedWeightConv3d(
            in_channels=dim,
            out_channels=dim,
            latent_size=latent_size,
            kernel_size=3,
            padding=1,
            bias=True,
            use_learned_mask=True
        )
        self.conv2 = AdaptiveSharedWeightConv3d(
            in_channels=dim,
            out_channels=dim,
            latent_size=latent_size,
            kernel_size=3,
            padding=1,
            bias=True,
            use_learned_mask=True
        )

    def forward(self, x, dlatents_in_slice):
        """
        x: [N, dim, D, H, W]
        dlatents_in_slice: [N, latent_size]
        returns: (out, mask2) # mask2: [N,1,D,H,W]
        """
        y, mask1 = self.conv1(x, dlatents_in_slice)
        y = self.act(y)
        y, mask2 = self.conv2(y, dlatents_in_slice)
        out = x + y
        return out, (mask1 + mask2)/2




# ------------------------------------------------------------------
# 5. 修改后的 transfer_model
#    - 在 forward 方法里添加 return_mask=False
#    - 如果 True，则收集每一层的 mask 并返回（3D mask 在 D 维做 mean 后得到 [N,1,H,W]）
#    - 如果 False，则只返回特征
# ------------------------------------------------------------------
class transfer_model(nn.Module):
    def __init__(self, latent_dim=512, n_blocks=4):
        super(transfer_model, self).__init__()
        activation = nn.ReLU(True)

        # 3D in
        BN_in = []
        for i in range(3):
            BN_in += [
                ResnetBlock_Adaptive3D(dim=32, latent_size=latent_dim, activation=activation)
            ]
        self.BottleNeck_3din = nn.Sequential(*BN_in)

        # 2D
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adaptive2D(dim=512, latent_size=latent_dim, activation=activation)
            ]
        self.BottleNeck_2d = nn.Sequential(*BN)

        # 3D out
        BN_out = []
        for i in range(3):
            BN_out += [
                ResnetBlock_Adaptive3D(dim=32, latent_size=latent_dim, activation=activation)
            ]
        self.BottleNeck_3dout = nn.Sequential(*BN_out)

        # 末端的普通 3D ResBlock (不产生 mask)
        self.resblocks_3d = nn.Sequential()
        for i in range(3):
            self.resblocks_3d.add_module(
                '3dr' + str(i),
                ResBlock3d(32, kernel_size=3, padding=1)
            )

    def forward(self, x, dlatents, return_mask=False):
        """
        x => [N, 32, D, H, W]
        dlatents => [N, latent_dim]
        return_mask => bool,
            True  => 返回 (输出, [mask_2d_1, mask_2d_2, ...])  # 每层一个 [N,1,H,W]
            False => 只返回输出 (默认)
        """
        # 如果需要收集 mask，则开一个 list
        mask_list = [] if return_mask else None

        # 1) 3D in
        for i in range(len(self.BottleNeck_3din)):
            x, mask_3d = self.BottleNeck_3din[i](x, dlatents)  # => ( [N,32,D,H,W], [N,1,D,H,W] )
            # 对 3D mask 在 D 维做 mean => [N,1,H,W]
            mask_2d = mask_3d.mean(dim=2, keepdim=False)       # => [N,1,H,W]
            if return_mask:
                mask_list.append(mask_2d)

        # 2) reshape to 2D => [N, 32*D, H, W]
        bs, c, d, h, w = x.shape
        x = x.view(bs, c*d, h, w)  # => [N, 32*D, H, W]

        # 2D blocks
        for i in range(len(self.BottleNeck_2d)):
            x, mask_2d = self.BottleNeck_2d[i](x, dlatents)  # => ( [N,32*D,H,W], [N,1,H,W] )
            if return_mask:
                mask_list.append(mask_2d)

        # reshape back => [N, 32, D, H, W]
        x = x.view(bs, c, d, h, w)  # => [N, 32, D, H, W]

        # 3) 3D out
        for i in range(len(self.BottleNeck_3dout)):
            x, mask_3d = self.BottleNeck_3dout[i](x, dlatents)  # => ( [N,32,D,H,W], [N,1,D,H,W] )
            mask_2d = mask_3d.mean(dim=2, keepdim=False)        # => [N,1,H,W]
            if return_mask:
                mask_list.append(mask_2d)

        # 4) 末端普通3D残差 (无mask)
        x = self.resblocks_3d(x)  # => [N, 32, D, H, W]

        if return_mask:
            # 返回 (特征, [mask_2d_1, mask_2d_2, ...])
            return x, mask_list
        else:
            # 只返回特征
            return x


class transfer_model2(nn.Module):
    def __init__(self, latent_dim=512, n_blocks=7):
        super(transfer_model2, self).__init__()
        activation = nn.ReLU(True)

        # # 3D in
        # BN_in = []
        # for i in range(3):
        #     BN_in += [
        #         ResnetBlock_Adaptive3D(dim=32, latent_size=latent_dim, activation=activation)
        #     ]
        # self.BottleNeck_3din = nn.Sequential(*BN_in)

        # 2D
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adaptive2D(dim=512, latent_size=latent_dim, activation=activation)
            ]
        self.BottleNeck_2d = nn.Sequential(*BN)

        # 3D out
        # BN_out = []
        # for i in range(3):
        #     BN_out += [
        #         ResnetBlock_Adaptive3D(dim=32, latent_size=latent_dim, activation=activation)
        #     ]
        # self.BottleNeck_3dout = nn.Sequential(*BN_out)

        # 末端的普通 3D ResBlock (不产生 mask)
        self.resblocks_3d = nn.Sequential()
        for i in range(6):
            self.resblocks_3d.add_module(
                '3dr' + str(i),
                ResBlock3d(32, kernel_size=3, padding=1)
            )

    def forward(self, x, dlatents, return_mask=False):
        """
        x => [N, 32, D, H, W]
        dlatents => [N, latent_dim]
        return_mask => bool,
            True  => 返回 (输出, [mask_2d_1, mask_2d_2, ...])  # 每层一个 [N,1,H,W]
            False => 只返回输出 (默认)
        """
        # 如果需要收集 mask，则开一个 list
        mask_list = [] if return_mask else None

        # 2) reshape to 2D => [N, 32*D, H, W]
        bs, c, d, h, w = x.shape
        x = x.view(bs, c*d, h, w)  # => [N, 32*D, H, W]

        # 2D blocks
        for i in range(len(self.BottleNeck_2d)):
            x, mask_2d = self.BottleNeck_2d[i](x, dlatents)  # => ( [N,32*D,H,W], [N,1,H,W] )
            if return_mask:
                mask_list.append(mask_2d)

        # reshape back => [N, 32, D, H, W]
        x = x.view(bs, c, d, h, w)  # => [N, 32, D, H, W]

        # 4) 末端普通3D残差 (无mask)
        x = self.resblocks_3d(x)  # => [N, 32, D, H, W]

        if return_mask:
            # 返回 (特征, [mask_2d_1, mask_2d_2, ...])
            return x, mask_list
        else:
            # 只返回特征
            return x

class transfer_model3(nn.Module):
    def __init__(self, latent_dim=512, n_blocks=7):
        super(transfer_model3, self).__init__()
        activation = nn.ReLU(True)

        # # 3D in
        # BN_in = []
        # for i in range(3):
        #     BN_in += [
        #         ResnetBlock_Adaptive3D(dim=32, latent_size=latent_dim, activation=activation)
        #     ]
        # self.BottleNeck_3din = nn.Sequential(*BN_in)

        # 2D
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adaptive2D(dim=512, latent_size=latent_dim, activation=activation)
            ]
        self.BottleNeck_2d = nn.Sequential(*BN)

        # 3D out
        # BN_out = []
        # for i in range(3):
        #     BN_out += [
        #         ResnetBlock_Adaptive3D(dim=32, latent_size=latent_dim, activation=activation)
        #     ]
        # self.BottleNeck_3dout = nn.Sequential(*BN_out)

        # 末端的普通 3D ResBlock (不产生 mask)
        self.resblocks_3d = nn.Sequential()
        for i in range(6):
            self.resblocks_3d.add_module(
                '3dr' + str(i),
                ResBlock3d(32, kernel_size=3, padding=1)
            )
        self.fc = nn.Linear(256*7*7, 512)

    def forward(self, x, dlatents, dlatents2,return_mask=False):
        """
        x => [N, 32, D, H, W]
        dlatents => [N, latent_dim]
        return_mask => bool,
            True  => 返回 (输出, [mask_2d_1, mask_2d_2, ...])  # 每层一个 [N,1,H,W]
            False => 只返回输出 (默认)
        """
        # 如果需要收集 mask，则开一个 list
        mask_list = [] if return_mask else None

        # 2) reshape to 2D => [N, 32*D, H, W]
        bs, c, d, h, w = x.shape
        x = x.view(bs, c*d, h, w)  # => [N, 32*D, H, W]
        dlatents2 = self.fc(dlatents2)
        # 2D blocks
        for i in range(len(self.BottleNeck_2d)):
            x, mask_2d = self.BottleNeck_2d[i](x, dlatents, dlatents2)  # => ( [N,32*D,H,W], [N,1,H,W] )
            if return_mask:
                mask_list.append(mask_2d)

        # reshape back => [N, 32, D, H, W]
        x = x.view(bs, c, d, h, w)  # => [N, 32, D, H, W]

        # 4) 末端普通3D残差 (无mask)
        x = self.resblocks_3d(x)  # => [N, 32, D, H, W]

        if return_mask:
            # 返回 (特征, [mask_2d_1, mask_2d_2, ...])
            return x, mask_list
        else:
            # 只返回特征
            return x
# if __name__ == "__main__":
#     # 假设 x 大小: N=2, C=32, D=4, H=64, W=64
#     # x = torch.randn(2, 32, 16, 64, 64)
#     # dlatents = torch.randn(2, 512)  # [N, latent_dim=512]

#     net = transfer_model(latent_dim=512, n_blocks=4)
#     # out = net(x, dlatents)
#     # print("out.shape:", out.shape)
#     total_params = sum(p.numel() for p in net.parameters())
#     print("total parameters:", total_params)
#     # 预期 [2, 32, 4, 64, 64]

# class G3d(nn.Module):
#     def __init__(self):
#         super(G3d, self).__init__()
#         # 下采样路径
#         self.downsampling = nn.Sequential(
#             ResBlock3D_stage3(32, 64),                      # [B, 32, 16, 64, 64] -> [B, 64, 16, 64, 64]
#             nn.AvgPool3d(kernel_size=2, stride=2),   # -> [B, 64, 8, 32, 32]
#             ResBlock3D_stage3(64, 128),                     # -> [B, 128, 8, 32, 32]
#             nn.AvgPool3d(kernel_size=2, stride=2),   # -> [B, 128, 4, 16, 16]
#             ResBlock3D_stage3(128, 256),                    # -> [B, 256, 4, 16, 16]
#             nn.AvgPool3d(kernel_size=2, stride=2),   # -> [B, 256, 2, 8, 8]
#             ResBlock3D_stage3(256, 512),                    # -> [B, 512, 2, 8, 8]
#         )

#         # 上采样路径
#         self.upsampling = nn.Sequential(
#             ResBlock3D_stage3(512, 256),                    # -> [B, 256, 2, 8, 8]
#             nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),  # -> [B, 256, 4, 16, 16]
#             ResBlock3D_stage3(256, 128),                    # -> [B, 128, 4, 16, 16]
#             nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),  # -> [B, 128, 8, 32, 32]
#             ResBlock3D_stage3(128, 64),                     # -> [B, 64, 8, 32, 32]
#             nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),  # -> [B, 64, 16, 64, 64]
#             ResBlock3D_stage3(64, 32), # -> [B, 32, 16, 64, 64]
#         )

#         # 最终输出层，将通道数恢复为32
#         self.final_conv = nn.Sequential(
#             nn.Conv3d(32, 32, kernel_size=3, padding=1)  # 保持最终维度
#         )

#     def forward(self, x):
#         """
#         输入: [B, 32, 16, 64, 64]
#         输出: [B, 32, 16, 64, 64]
#         """
#         x = self.downsampling(x)
#         x = self.upsampling(x)
#         x = self.final_conv(x)
#         return x

# class G3d(nn.Module):
#     def __init__(self):
#         super(G3d, self).__init__()
#         # 简单串联几个3D ResBlock
#         self.resblocks = nn.Sequential(
#             ResBlock3D_stage3(32, 32),  # 保持通道数不变
#             ResBlock3D_stage3(32, 32),
#             ResBlock3D_stage3(32, 32),
#             ResBlock3D_stage3(32, 32),
#             ResBlock3D_stage3(32, 32),
#             ResBlock3D_stage3(32, 32)
#         )

#     def forward(self, x):
#         """
#         输入: [B, 32, 16, 64, 64]
#         输出: [B, 32, 16, 64, 64]
#         """
#         return self.resblocks(x)


class G3d(nn.Module):
    def __init__(self):
        super(G3d, self).__init__()
        # 简单串联几个3D ResBlock
        self.resblocks1 = nn.Sequential(
            ResBlock3D_stage3_leak(32, 32),  # 保持通道数不变
            ResBlock3D_stage3_leak(32, 32),
            ResBlock3D_stage3_leak(32, 32)
        )
        self.resblocks2 = nn.Sequential(
            ResBlock2d(32*16, 32*16, kernel_size=(3, 3), padding=(1, 1)),  # 保持通道数不变
            ResBlock2d(32*16, 32*16, kernel_size=(3, 3), padding=(1, 1)),
            ResBlock2d(32*16, 32*16, kernel_size=(3, 3), padding=(1, 1))
        )
        self.resblocks3 = nn.Sequential(
            ResBlock3D_stage3_leak(32, 32),  # 保持通道数不变
            ResBlock3D_stage3_leak(32, 32),
            ResBlock3D_stage3_leak(32, 32)
        )
        # self.third = SameBlock2d(32*16, 32*16, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        # self.fourth = nn.Conv2d(in_channels=32*16, out_channels=32*16, kernel_size=1, stride=1)
    def forward(self, x):
        """
        输入: [B, 32, 16, 64, 64]
        输出: [B, 32, 16, 64, 64]
        """
        x = self.resblocks1(x)
        bs, c, d, h, w = x.shape
        x = x.view(bs, c*d, h, w)  # => [N, 32*D, H, W]
        x = self.resblocks2(x)
        # x = self.fourth(x)
        x = x.view(bs, c, d, h, w)  # => [N, C, D, H, W]
        x = self.resblocks3(x)
        return x

# class G3d(nn.Module):
#     def __init__(self):
#         super(G3d, self).__init__()
#         # 下采样路径
#         self.down1 = ResBlock3D_stage3(32, 64)                      # [B, 32, 16, 64, 64] -> [B, 64, 16, 64, 64]
#         self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)   # -> [B, 64, 8, 32, 32]
#         self.down2 = ResBlock3D_stage3(64, 128)                     # -> [B, 128, 8, 32, 32]
#         self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)   # -> [B, 128, 4, 16, 16]
#         self.down3 = ResBlock3D_stage3(128, 256)                    # -> [B, 256, 4, 16, 16]
#         self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2)   # -> [B, 256, 2, 8, 8]
#         self.down4 = ResBlock3D_stage3(256, 512)                    # -> [B, 512, 2, 8, 8]

#         # 上采样路径
#         self.up1 = ResBlock3D_stage3(512, 256)                    # [B, 512, 2, 8, 8] -> [B, 256, 2, 8, 8]
#         self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)  # -> [B, 256, 4, 16, 16]

#         self.up2 = ResBlock3D_stage3(512, 128)                    # [B, 512(256+256), 4, 16, 16] -> [B, 128, 4, 16, 16]
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)  # -> [B, 128, 8, 32, 32]

#         self.up3 = ResBlock3D_stage3(256, 64)                     # [B, 256(128+128), 8, 32, 32] -> [B, 64, 8, 32, 32]
#         self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)  # -> [B, 64, 16, 64, 64]

#         self.up4 = ResBlock3D_stage3(128, 32)                     # [B, 128(64+64), 16, 64, 64] -> [B, 32, 16, 64, 64]

#         # 最终输出层
#         self.final_conv = nn.Conv3d(32, 32, kernel_size=3, padding=1)

#     def forward(self, x):
#         # 保存初始输入
#         x_input = x  # [B, 32, 16, 64, 64]

#         # 下采样路径
#         x1 = self.down1(x)                    # [B, 64, 16, 64, 64]
#         p1 = self.pool1(x1)                   # [B, 64, 8, 32, 32]

#         x2 = self.down2(p1)                   # [B, 128, 8, 32, 32]
#         p2 = self.pool2(x2)                   # [B, 128, 4, 16, 16]

#         x3 = self.down3(p2)                   # [B, 256, 4, 16, 16]
#         p3 = self.pool3(x3)                   # [B, 256, 2, 8, 8]

#         x4 = self.down4(p3)                   # [B, 512, 2, 8, 8]

#         # 上采样路径
#         u1 = self.up1(x4)                     # [B, 256, 2, 8, 8]
#         u1 = self.upsample1(u1)               # [B, 256, 4, 16, 16]
#         u1 = torch.cat([u1, x3], dim=1)       # [B, 512, 4, 16, 16]

#         u2 = self.up2(u1)                     # [B, 128, 4, 16, 16]
#         u2 = self.upsample2(u2)               # [B, 128, 8, 32, 32]
#         u2 = torch.cat([u2, x2], dim=1)       # [B, 256, 8, 32, 32]

#         u3 = self.up3(u2)                     # [B, 64, 8, 32, 32]
#         u3 = self.upsample3(u3)               # [B, 64, 16, 64, 64]
#         u3 = torch.cat([u3, x1], dim=1)       # [B, 128, 16, 64, 64]

#         u4 = self.up4(u3)                     # [B, 32, 16, 64, 64]

#         output = self.final_conv(u4)          # [B, 32, 16, 64, 64]

#         return output

# ----------------  测试运行  ----------------
if __name__ == "__main__":
    # 随机输入测试
    N, C, D, H, W = 2, 32, 16, 64, 64
    latent_dim = 512

    x = torch.randn(N, C, D, H, W)
    dlatents = torch.randn(N, latent_dim)

    model = transfer_model(latent_dim=latent_dim, n_blocks=4)

    # 1) return_mask = False，只返回最终特征
    out_no_mask = model(x, dlatents, return_mask=False)
    print("out_no_mask shape:", out_no_mask.shape)

    # 2) return_mask = True，同时收集所有 mask
    out_with_mask, mask_list = model(x, dlatents, return_mask=True)
    print("out_with_mask shape:", out_with_mask.shape)
    print(f"mask_list len: {len(mask_list)}")
    for i, m in enumerate(mask_list):
        print(f"Mask {i} shape = {m.shape}")
