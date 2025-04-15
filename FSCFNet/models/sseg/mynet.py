import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sseg.ffc2 import FFC_To_CloudNet
from models.sseg.pvtv2 import pvt_v2_b2, OverlapPatchEmbed
import numpy as np
import math
from thop import profile
from typing import List
from models.sseg.dsam import Adaptive_DSAM
from pytorch_wavelets import DTCWTForward, DTCWTInverse

class BasicConv2d(nn.Module):   #conv+bn+relu
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):    #转置卷积+bn+relu
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1,output_padding=0, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,output_padding= output_padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.inch = in_planes
    def forward(self, x):

        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class FFT(nn.Module):
    def __init__(self,inchannel,outchannel):
        super().__init__()
        self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        # self.DWT =DTCWTForward(J=3, include_scale=True)
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.conv1 = BasicConv2d(outchannel, outchannel)
        self.conv2 = BasicConv2d(inchannel, outchannel)
        self.conv3 = BasicConv2d(outchannel, outchannel)
        self.change = TransBasicConv2d(outchannel, outchannel)

    def forward(self, x, y):
        y = self.conv2(y)
        Xl, Xh = self.DWT(x)
        Yl, Yh = self.DWT(y)
        x_y = self.conv1(Xl) + self.conv1(Yl)

        x_m = self.IWT((x_y, Xh))
        y_m = self.IWT((x_y, Yh))

        out = self.conv3(x_m + y_m)
        return out





def equalize_tensor(tensor):
    """
    将一个 (1, 1, 384, 384) 张量中的值均衡化到 0-65535 之间。

    Args:
      tensor: 一个 (1, 1, 384, 384) 的张量。

    Returns:
      一个 (1, 1, 384, 384) 的张量，其中所有元素的值都在 0-65535 之间。
    """
    # 计算最小值和最大值
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)

    scaled_tensor = (tensor - min_value) * (65535 / (max_value - min_value))

    return scaled_tensor


class Curvature(torch.nn.Module):
    def __init__(self, ratio):
        super(Curvature, self).__init__()
        weights = torch.tensor([[[[-1 / 16, 5 / 16, -1 / 16], [5 / 16, -1, 5 / 16], [-1 / 16, 5 / 16, -1 / 16]]]])
        self.weight = torch.nn.Parameter(weights).cuda()
        self.ratio = ratio

    def forward(self, x):
        B, C, H, W = x.size()
        x_origin = x
        x = x.reshape(B * C, 1, H, W)
        out = F.conv2d(x, self.weight)
        out = torch.abs(out)
        p = torch.sum(out, dim=-1)
        p = torch.sum(p, dim=-1)
        p = p.reshape(B, C)

        _, index = torch.topk(p, int(self.ratio * C), dim=1)
        selected = []
        for i in range(x_origin.shape[0]):
            selected.append(torch.index_select(x_origin[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)

        return selected


class Entropy_Hist(nn.Module):
    def __init__(self, ratio, win_w=3, win_h=3):
        super(Entropy_Hist, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.ratio = ratio

    def calcIJ_new(self, img_patch):
        total_p = img_patch.shape[-1] * img_patch.shape[-2]
        if total_p % 2 != 0:
            tem = torch.flatten(img_patch, start_dim=-2, end_dim=-1)
            center_p = tem[:, :, :, int(total_p / 2)]
            mean_p = (torch.sum(tem, dim=-1) - center_p) / (total_p - 1)
            if torch.is_tensor(img_patch):
                return center_p * 100 + mean_p
            else:
                return (center_p, mean_p)
        else:
            print("modify patch size")

    def histc_fork(ij):
        BINS = 256
        B, C = ij.shape
        N = 16
        BB = B // N
        min_elem = ij.min()
        max_elem = ij.max()
        ij = ij.view(N, BB, C)

        def f(x):
            with torch.no_grad():
                res = []
                for e in x:
                    res.append(torch.histc(e, bins=BINS, min=min_elem, max=max_elem))
                return res

        futures: List[torch.jit.Future[torch.Tensor]] = []

        for i in range(N):
            futures.append(torch.jit.fork(f, ij[i]))

        results = []
        for future in futures:
            results += torch.jit.wait(future)
        with torch.no_grad():
            out = torch.stack(results)
        return out

    def forward(self, img):
        with torch.no_grad():
            B, C, H, W = img.shape
            ext_x = int(self.win_w / 2)  # 考虑滑动窗口大小，对原图进行扩边，扩展部分长度
            ext_y = int(self.win_h / 2)

            new_width = ext_x + W + ext_x  # 新的图像尺寸
            new_height = ext_y + H + ext_y

            # 使用nn.Unfold依次获取每个滑动窗口的内容
            nn_Unfold = nn.Unfold(kernel_size=(self.win_w, self.win_h), dilation=1, padding=ext_x, stride=1)
            # 能够获取到patch_img，shape=(B,C*K*K,L),L代表的是将每张图片由滑动窗口分割成多少块---->28*28的图像，3*3的滑动窗口，分成了28*28=784块
            x = nn_Unfold(img)  # (B,C*K*K,L)
            x = x.view(B, C, 3, 3, -1).permute(0, 1, 4, 2, 3)  # (B,C*K*K,L) ---> (B,C,L,K,K)
            ij = self.calcIJ_new(x).reshape(B * C,
                                            -1)  # 计算滑动窗口内中心的灰度值和窗口内除了中心像素的灰度均值,(B,C,L,K,K)---> (B,C,L) ---> (B*C,L)

            fij_packed = self.histc_fork(ij)
            p = fij_packed / (new_width * new_height)
            h_tem = -p * torch.log(torch.clamp(p, min=1e-40)) / math.log(2)

            a = torch.sum(h_tem, dim=1)  # 对所有二维熵求和，得到这张图的二维熵
            H = a.reshape(B, C)

            _, index = torch.topk(H, int(self.ratio * C), dim=1)  # Nx3
        selected = []
        for i in range(img.shape[0]):
            selected.append(torch.index_select(img[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)

        return selected


# 通道变换函数，这里使用简单的加权平均
def channel_transform(input_data):
    # 定义权重，这里假设所有通道的权重相等
    weights = np.ones(4) / 4.0

    # 计算加权平均，将4通道映射到3通道
    transformed_data = np.sum(input_data * weights[:, np.newaxis, np.newaxis], axis=0)

    return transformed_data


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            # 计算为570 = ((572 + 2*padding - dilation*(kernal_size-1) -1) / stride ) +1
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# 实现右边的向上的采样操作，并完成该层相应的卷积操作。由于此操作是完成跳级结构和unet网络的级联，所以在此处，输入输出的ch应该为解码器输出和跳级结构的通道数之和
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            # 声明使用的上采样方法为bilinear——双线性插值,默认使用这个值,
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 否则就使用转置卷积来实现上采样,计算式子为 （Height-1）*stride - 2*padding
            self.up = nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):  # x2是左边特征提取传来的值
        # 第一次上采样返回56*56，但是还没结束
        # x1 = (batch_size, channel, H, W)

        # input is CHW [0]是batch_size, [1]是通道数,更改了下,与源码不同
        diffY = x2.size()[2] - x1.size()[2]  # 得到图像x2与x1的H的差值，64-56=8
        diffX = x2.size()[3] - x1.size()[3]  # 得到图像x2与x1的W差值

        # 用第一次上采样为例,即当上采样后的结果大小与右边的特征的结果大小不同时，通过填充来使x2的大小与x1相同
        # 对图像进行填充(4,4,4,4),左右上下都增加4，所以最后使得56*56变为64*64
        x1 = F.pad(x1, [int(diffX // 2), int(diffX - diffX // 2), int(diffY // 2), int(diffY - diffY // 2)], 'constant')
        # TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect.
        # 将最后上采样得到的值x1和左边特征提取的值进行拼接,dim=1即在通道数上进行拼接，由512变为1024
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.up(x)
        return x


# 实现右边的最高层的最右边的卷积
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Inconv(nn.Module):
    '''(conv => BN => ReLU)'''

    def __init__(self, in_ch, out_ch):
        super(Inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            # 以第一层为例进行讲解
            # 输入通道数in_ch,输出通道数out_ch,卷积核设为kernal_size 3*3,
            # 计算为570 = ((572 + 2*padding - dilation*(kernal_size-1) -1) / stride ) +1
            nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=1),
            # 进行批标准化，在训练时，该层计算每次输入的均值与方差，并进行移动平均
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Triple_conv(nn.Module):
    '''(conv => BN => ReLU) * 3'''

    def __init__(self, in_ch, out_ch):
        super(Triple_conv, self).__init__()
        self.conv = nn.Sequential(
            # 以第一层为例进行讲解
            # 输入通道数in_ch,输出通道数out_ch,卷积核设为kernal_size 3*3,
            # 计算为570 = ((572 + 2*padding - dilation*(kernal_size-1) -1) / stride ) +1
            nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=1),
            # 进行批标准化，在训练时，该层计算每次输入的均值与方差，并进行移动平均
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Contr_arm(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Contr_arm, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 1, padding=0, dilation=1),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True)
        )
        self.conv_d = Double_conv(in_ch, out_ch)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = torch.cat([x, x1], dim=1)
        x2 = self.conv_d(x)
        x = x1 + x2
        x = self.relu(x)
        xout = self.maxpool(x)
        return x, xout


class Imprv_contr_arm(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Imprv_contr_arm, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 1, padding=0, dilation=1),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, padding=0, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_d = Double_conv(in_ch, out_ch)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = torch.cat([x, x1], dim=1)
        x2 = self.conv_d(x)
        x3 = self.conv3(x2)
        x2 = self.conv2(x2)
        x = x1 + x2 + x3
        x = self.relu(x)
        xout = self.maxpool(x)
        return x, xout


class Bridge(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Bridge, self).__init__()
        self.conv_d = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15)
        )
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x1 = y
        x1 = torch.cat([x, x1], dim=1)
        x2 = self.conv_d(x)
        x = x1 + x2
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Improve_ff_block4(nn.Module):
    def __init__(self):
        super(Improve_ff_block4, self).__init__()
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(4)
        self.maxpool8 = nn.MaxPool2d(8)
        self.maxpool16 = nn.MaxPool2d(16)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor1, input_tensor2, input_tensor3, input_tensor4, pure_ff):
        for xi in range(1):
            if xi == 0:
                x1 = input_tensor1
            x1 = torch.cat([x1, input_tensor1], dim=1)
        x1 = self.maxpool2(x1)

        for xi in range(3):
            if xi == 0:
                x2 = input_tensor2
            x2 = torch.cat([x2, input_tensor2], dim=1)
        x2 = self.maxpool4(x2)

        for xi in range(7):
            if xi == 0:
                x3 = input_tensor3
            x3 = torch.cat([x3, input_tensor3], dim=1)
        x3 = self.maxpool8(x3)

        for xi in range(15):
            if xi == 0:
                x4 = input_tensor4
            x4 = torch.cat([x4, input_tensor4], dim=1)
        x4 = self.maxpool16(x4)
        x = x1 + x2 + x3 + x4 + pure_ff
        x = self.relu(x)
        return x


class Improve_ff_block3(nn.Module):
    def __init__(self):
        super(Improve_ff_block3, self).__init__()
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(4)
        self.maxpool8 = nn.MaxPool2d(8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor1, input_tensor2, input_tensor3, pure_ff):
        for xi in range(1):
            if xi == 0:
                x1 = input_tensor1
            x1 = torch.cat([x1, input_tensor1], dim=1)
        x1 = self.maxpool2(x1)

        for xi in range(3):
            if xi == 0:
                x2 = input_tensor2
            x2 = torch.cat([x2, input_tensor2], dim=1)
        x2 = self.maxpool4(x2)

        for xi in range(7):
            if xi == 0:
                x3 = input_tensor3
            x3 = torch.cat([x3, input_tensor3], dim=1)
        x3 = self.maxpool8(x3)
        x = x1 + x2 + x3 + pure_ff
        x = self.relu(x)
        return x


class Improve_ff_block2(nn.Module):
    def __init__(self):
        super(Improve_ff_block2, self).__init__()
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool4 = nn.MaxPool2d(4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor1, input_tensor2, pure_ff):
        for xi in range(1):
            if xi == 0:
                x1 = input_tensor1
            x1 = torch.cat([x1, input_tensor1], dim=1)
        x1 = self.maxpool2(x1)

        for xi in range(3):
            if xi == 0:
                x2 = input_tensor2
            x2 = torch.cat([x2, input_tensor2], dim=1)
        x2 = self.maxpool4(x2)
        x = x1 + x2 + pure_ff
        x = self.relu(x)
        return x


class Improve_ff_block1(nn.Module):
    def __init__(self):
        super(Improve_ff_block1, self).__init__()
        self.maxpool2 = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor1, pure_ff):
        for xi in range(1):
            if xi == 0:
                x1 = input_tensor1
            x1 = torch.cat([x1, input_tensor1], dim=1)
        x1 = self.maxpool2(x1)
        x = x1 + pure_ff
        x = self.relu(x)
        return x


class Cloudnet(nn.Module):
    def __init__(self, n_channels, n_classes, mode='curvature'):
        super(Cloudnet, self).__init__()

        self.mode = mode
        self.backbone = pvt_v2_b2(in_chans=4)  # [64, 128, 320, 512]
        path = '/media/estar/Data/HWJ/MCDNet-main/models/sseg/pvt_v2_b2.pth'
        save_model = torch.load(path)
        pretrained_model = save_model
        original_weights = save_model['patch_embed1.proj.weight']
        # 扩展通道数为4，创建一个新的权重参数
        expanded_weights = torch.cat(
            [original_weights, torch.zeros(original_weights.shape[0], 1, *original_weights.shape[2:])], dim=1)
        # 更新模型的权重参数
        save_model.patch_embed1 = OverlapPatchEmbed(in_chans=4)
        save_model['patch_embed1.proj.weight'] = expanded_weights
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.backbone2 = pvt_v2_b2(in_chans=4)
        model_dict2 = self.backbone2.state_dict()
        state_dict2 = {k: v for k, v in save_model.items() if k in model_dict2.keys()}
        model_dict2.update(state_dict2)
        self.backbone2.load_state_dict(model_dict2)

        self.inconv = Inconv(n_channels, 16)
        self.conv1 = Contr_arm(16, 32)
        self.conv2 = Contr_arm(32, 64)
        self.conv3 = Contr_arm(64, 128)
        self.conv4 = Contr_arm(128, 256)
        self.conv5 = Imprv_contr_arm(256, 512)
        self.conv6 = Bridge(512, 1024)
        self.conv7 = nn.ConvTranspose2d(1024, 512, 2, stride=2)

        self.FFC_1 = FFC_To_CloudNet(dim=512, in_ch=512)
        self.FFC_2 = FFC_To_CloudNet(dim=320, in_ch=320)
        self.FFC_3 = FFC_To_CloudNet(dim=128, in_ch=128)
        self.FFC_4 = FFC_To_CloudNet(dim=64, in_ch=64)

        self.S2 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(320, 1, 3, stride=1, padding=1)
        self.S4 = nn.Conv2d(128, 1, 3, stride=1, padding=1)
        self.S5 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up1 = up(1024, 320)
        self.up2 = up(640, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 16)
        self.outc = outconv(16, n_classes)

        if self.mode == 'ori':
            self.ratio = [0, 0]
        if self.mode == 'curvature':
            self.ife1 = Curvature(0.6)
            self.ife2 = Curvature(0.5)

        if self.mode == 'entropy':
            self.ife1 = Entropy_Hist(0.6)
            self.ife2 = Entropy_Hist(0.5)

        self.conv_1 = nn.Conv2d(320, 320, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.up_1 = nn.MaxPool2d(2)
        self.up_2 = nn.MaxPool2d(2)

        #self.Adaptive = Adaptive_DSAM(3)
        self.adpconv = nn.Conv2d(1, 4, 3, stride=1, padding=1)
        self.adpconv2 = nn.Conv2d(3, 4, 3, stride=1, padding=1)

        self.conv8 = nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=1)
        self.conv9 = nn.ConvTranspose2d(640, 320, 3, stride=1, padding=1)
        self.conv10 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1)
        self.conv11 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1)

        self.FFT = FFT(4, 4)

    def forward(self, x, rgb, y, z):
        # x = x.unsqueeze(1)

        y = y - z
        y = self.adpconv(y.float())
        rgb = self.adpconv2(rgb.float())

        m = self.FFT(rgb, y)

        s = self.backbone2(m)
        pvt = self.backbone(x)
        m1 = s[0]
        m2 = s[1]
        m3 = s[2]
        m4 = s[3]

        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x5 = self.conv6(x4, m4)
        x6 = self.conv7(x5)

        x7p = torch.cat([x4, m4], dim=1)  # 512->1024
        x8p = torch.cat([x3, m3], dim=1)  # 320->640
        x9p = torch.cat([x2, m2], dim=1)  # 128->256
        x10p = torch.cat([x1, m1], dim=1)  # 64->128

        x7p = self.conv8(x7p)  # 1024->512
        x8p = self.conv9(x8p)  # 640->320
        x9p = self.conv10(x9p)  # 256->128
        x10p = self.conv11(x10p)  # 128->64

        x9_p = self.up_1(x9p)
        x10_p = self.up_2(x10p)

        x8p = torch.cat([self.ife1(x8p), x9_p], dim=1)
        x9p = torch.cat([self.ife2(x9p), x10_p], dim=1)
        x8p = x3 + x8p
        x9p = x2 + x9p

        x8p = self.conv_1(x8p)
        x9p = self.conv_2(x9p)

        s2 = x6

        s3 = self.up1(s2, x7p)
        s4 = self.up2(s3, x8p)
        s5 = self.up3(s4, x9p)
        s6 = self.up4(s5, x10p)

        out = self.outc(s6)
        out = self.upsample2(out)

        s3 = self.S3(s3)
        s3 = self.upsample16(s3)
        s4 = self.S4(s4)
        s4 = self.upsample8(s4)
        s5 = self.S5(s5)
        s5 = self.upsample4(s5)

        return torch.sigmoid(out), torch.sigmoid(s3), torch.sigmoid(s4), torch.sigmoid(s5)


import time
def measure_latency_and_fps(model, input1,input2,input3,input4, num_iterations=100):
    # 将模型移到GPU（如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    input1 = input1.to(device)
    input2 = input2.to(device)
    input3 = input3.to(device)
    input4 = input4.to(device)


    # 测量延迟
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(input1,input2,input3,input4)
    end_time = time.time()

    # 计算单次延迟（毫秒）
    total_time = end_time - start_time
    average_latency = (total_time / num_iterations) * 1000  # 转换为毫秒

    # 测量FPS
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(input1,input2,input3,input4)
    end_time = time.time()

    total_time_fps = end_time - start_time
    fps = num_iterations / total_time_fps

    return average_latency, fps

if __name__ == '__main__':
    import os
    import torch
    from thop import profile  # Make sure to import the profile function

    # Initialize the model
    net =Cloudnet(4,1)

    # Move the model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)

    # Create input tensors and move them to the same device
    input1 = torch.randn(1, 4, 384, 384).to(device)
    input2 = torch.randn(1, 3, 384, 384).to(device)
    input3 = torch.randn(1, 1, 384, 384).to(device)
    input4 = torch.randn(1, 1, 384, 384).to(device)

    # Profile the model
    flops, params = profile(net, (input1,input2,input3,input4))
    print('GFLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    latency, fps = measure_latency_and_fps(net, input1,input2,input3,input4)
    print(f'平均延迟: {latency:.2f} ms')
    print(f'FPS: {fps:.2f}')