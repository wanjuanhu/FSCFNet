import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc
from typing import Tuple, List

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
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
        res = x * y.expand_as(x)
        return res

def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class BaseDiscriminator(nn.Module):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Predict scores and get intermediate activations. Useful for feature matching loss
        :return tuple (scores, list of intermediate activations)
        """
        raise NotImplemented()





class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg #将通道分为两个部分，一部分作为全局一部分作为局部
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #自适应平均池化层 self.avgpool，将输入特征进行全局池化，输出尺寸为 (1, 1)。
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True) #输入通道channel，输出通道channel//16，1*1卷积核
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        # 根据局部特征通道数和全局特征通道数的值，创建了将全局特征映射为局部特征的 1x1 卷积层 self.conv_a2l（如果 in_cl 为 0，则为 None），
        # 以及将全局特征映射为全局特征的 1x1 卷积层 self.conv_a2g（如果 in_cg 为 0，则为 None）。创建了 Sigmoid 函数 self.sigmoid。

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x
        # 前向传播方法，接受输入 x。如果输入 x 不是元组类型，则将其转换为包含一个特征张量和 0 的元组 (x, 0)。将元组拆分为局部特征 id_l 和全局特征 id_g。
        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))
        # 根据全局特征 id_g 的类型，如果是整数，则只使用局部特征 id_l；否则，将局部特征和全局特征在通道维度上进行拼接。
        # 将特征张量 x 经过自适应平均池化层 self.avgpool 进行全局池化操作，并通过卷积层 self.conv1 和 ReLU 激活函数 self.relu1 进行特征变换。
        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g
    # 根据 self.conv_a2l 和 self.conv_a2g 的存在与否，对局部特征和全局特征进行调整。
    # 如果 self.conv_a2l 存在，则将局部特征 id_l 乘以经过 Sigmoid 函数映射后的 self.conv_a2l(x)；
    # 如果 self.conv_a2g 存在，则将全局特征 id_g 乘以经过 Sigmoid 函数映射后的 self.conv_a2g(x)。
    # 最后，返回调整后的局部特征 x_l 和全局特征 x_g。


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        # 设置类的属性 groups 为传入的分组数 groups。创建一个卷积层 self.conv_layer，
        # 输入通道数为 in_channels * 2 加上谱位置编码的通道数（如果启用了谱位置编码），
        # 输出通道数为 out_channels * 2，卷积核大小为 1x1，步幅为 1，填充为 0，分组数为 groups，
        # 且不使用偏置。创建一个批标准化层 self.bn 和 ReLU 激活函数 self.relu


        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)
        # 前向传播方法，接受输入张量 x。获取批次大小 batch。如果 spatial_scale_factor 不为 None，则保存原始大小，
        # 并将输入张量 x 进行插值操作，使其空间尺度缩放为 spatial_scale_factor 倍。

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        # 如果 self.ffc3d 为假，即不使用三维傅里叶变换，那么 fft_dim 的值被设置为 (-2, -1)。这表示在进行傅里叶变换时，将对最后两个维度进行变换。
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        # 获取输入张量 x 的大小。根据是否启用了 FFC3D，确定傅里叶变换的维度 fft_dim。
        # 使用 torch.fft.rfftn 对输入张量 x 进行实数域的快速傅里叶变换，指定变换维度为 fft_dim，
        # 归一化方式为 self.fft_norm。将实部和虚部通过 torch.stack 拼接为一个新的张量，
        # 维度为 (-3, -2, -1, 2)。通过 permute 和 contiguous 操作进行维度重排，
        # 得到维度为 (batch, c, 2, h, w/2+1) 的张量。最后，将张量形状变为 (batch, -1, h, w/2+1)。


        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)
        # 如果启用了谱位置编码 spectral_pos_encoding，则根据 FFT 变换后的张量 ffted 的形状获取高度和宽度，
        # 并生成相应的坐标张量。通过 torch.linspace 在高度和宽度上生成均匀的坐标值，扩展为与 ffted 相同的形状。
        # 将坐标张量与 ffted 拼接在一起，维度上增加 2。

        if self.use_se:
            ffted = self.se(ffted)
        # 如果启用了 SE 模块 use_se，则将 ffted 输入到 SE 模块 self.se 进行处理。

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        # 通过卷积层和批归一化层对频域表示进行特征变换和归一化操作，然后通过 ReLU 激活函数进行激活。

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        # 对输出进行形状调整，将频域表示的维度重新排列，以便进行逆傅里叶变换。然后，将实部和虚部组合为复数形式

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        # 根据是否使用三维傅里叶变换，选择逆傅里叶变换的维度 fft_dim，并指定逆傅里叶变换的输出形状为 ifft_shape_slice，使用给定的归一化方式 self.fft_norm 进行逆傅里叶变换。

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)
        # 如果进行了空间尺度缩放，将输出恢复到原始尺寸。

        return output


class SpectralTransform(nn.Module):
#用于频谱转化操作
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()
        #根据给定的stride值，选择使用平均池化（AvgPool2d）还是恒等映射（Identity）作为下采样层。

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        #如果enable_lfu为True，则创建另一个FourierUnit实例，并将其赋值给lfu属性。
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        # 如果enable_lfu为True，则对输入张量进行切片和LFU操作，并将结果复制和重复以与输出尺寸相匹配；否则将xs设置为0。

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g
        # ratio_gin：输入张量中全局信息通道的比例。
        # ratio_gout：输出张量中全局信息通道的比例。

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg
        # ratio_gin：输入张量中全局信息通道的比例。
        # ratio_gout：输出张量中全局信息通道的比例。

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)
            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
            #如果启用了门控机制，则将局部信息张量和全局信息张量拼接成总输入张量。然后通过门控层计算门控值，将其分为g2l_gate和l2g_gate。
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFC_To_CloudNet(nn.Module):
    def __init__(self, dim, in_ch, padding_type='reflect', norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=True, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, ratio_gin=0.5, ratio_gout=0.5,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = nn.Conv2d(in_ch//2, in_ch//2, 1, padding=0, dilation=1)
        self.inline = inline
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, dilation=1)

    def forward(self, x):
        x = self.conv(x)
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        x_l = self.conv2(x_l)
        x_g = self.conv2(x_g)
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out