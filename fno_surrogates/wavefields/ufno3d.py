'''
@author: Yang Cui, Uppsala University
Modified U-FNO for 3D seismic wavefield prediction
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce


# Spectral Convolution

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


# 3D U-NET BLOCK

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dropout_rate=0.0):
        super().__init__()
        self.conv1 = self.conv(input_channels, output_channels, kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2 = self.conv(output_channels, output_channels, kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2_1 = self.conv(output_channels, output_channels, kernel_size, stride=1, dropout_rate=dropout_rate)
        self.conv3 = self.conv(output_channels, output_channels, kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv3_1 = self.conv(output_channels, output_channels, kernel_size, stride=1, dropout_rate=dropout_rate)
        self.deconv2 = self.deconv(output_channels, output_channels)
        self.deconv1 = self.deconv(output_channels * 2, output_channels)
        self.deconv0 = self.deconv(output_channels * 2, output_channels)
        self.output_layer = nn.Conv3d(output_channels * 2, output_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), dim=1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), dim=1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), dim=1)
        out = self.output_layer(concat0)
        return out

    def conv(self, in_planes, out_planes, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )


# U-FNO BLOCK

class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels, out_channels):
        super().__init__()
        self.width = width
        # Input lifting
        self.fc0 = nn.Conv3d(in_channels, width, kernel_size=1)
        # Fourier layers
        self.conv0 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv1 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv2 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv3 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv4 = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv5 = SpectralConv3d(width, width, modes1, modes2, modes3)
        # Pointwise convs
        self.w0 = nn.Conv3d(width, width, 1)
        self.w1 = nn.Conv3d(width, width, 1)
        self.w2 = nn.Conv3d(width, width, 1)
        self.w3 = nn.Conv3d(width, width, 1)
        self.w4 = nn.Conv3d(width, width, 1)
        self.w5 = nn.Conv3d(width, width, 1)
        # U-NET refinements
        self.unet3 = U_net(width, width)
        self.unet4 = U_net(width, width)
        self.unet5 = U_net(width, width)
        # Output projection
        self.fc1 = nn.Conv3d(width, 128, 1)
        self.fc2 = nn.Conv3d(128, out_channels, 1)

    def forward(self, x):
        # x: [B, T, X, Y, Z]
        x = self.fc0(x)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        # U-FNO layers
        x1 = self.conv4(x)
        x2 = self.w4(x)
        x3 = self.unet4(x)
        x = F.gelu(x1 + x2 + x3)
        x1 = self.conv5(x)
        x2 = self.w5(x)
        x3 = self.unet5(x)
        x = F.gelu(x1 + x2 + x3)
        # Projection
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x