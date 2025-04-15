import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, bilinear=False):
        super(UNet, self).__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)

        return torch.sigmoid(x)


import time
def measure_latency_and_fps(model, input_tensor, num_iterations=100):
    # 将模型移到GPU（如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    input_tensor = input_tensor.to(device)

    # 测量延迟
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(input_tensor)
    end_time = time.time()

    # 计算单次延迟（毫秒）
    total_time = end_time - start_time
    average_latency = (total_time / num_iterations) * 1000  # 转换为毫秒

    # 测量FPS
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(input_tensor)
    end_time = time.time()

    total_time_fps = end_time - start_time
    fps = num_iterations / total_time_fps

    return average_latency, fps

if __name__ == '__main__':
    import os
    import torch
    from thop import profile  # Make sure to import the profile function

    # Initialize the model
    net = UNet(4,1)

    # Move the model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)

    # Create input tensors and move them to the same device
    input1 = torch.randn(1, 4, 384, 384).to(device)

    # Profile the model
    flops, params = profile(net, (input1,))
    print('GFLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    latency, fps = measure_latency_and_fps(net, input1)
    print(f'平均延迟: {latency:.2f} ms')
    print(f'FPS: {fps:.2f}')

    