# -*- coding: utf-8 -*-
# @Author  : C Zhou
# @File    : SPCNet.py
# @Software: PyCharm

import torch.nn as nn
import torch
import torch.nn.functional as F

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, n_block, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(n_block - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv(out)
        out = self.norm(out)
        return out

class ConvSnp_norm(nn.Module):
    " BN -> ReLU -> convolution1×1 -> BN -> ReLU -> convolution3×3 -> concat res -> "
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvSnp_norm, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(2*in_channels)
        self.activation = get_activation(activation)
        self.conv_out = nn.Conv2d(2*in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = x
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv3(out)

        x1 = self.bn2(x1)
        x1 = self.activation(x1)
        x1 = self.conv1(x1)

        out = torch.cat([x1, out], dim=1)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv_out(out)
        return out

def _make_nConvSnp(in_channels, out_channels, n_block, activation='ReLU'):
    layers = []
    layers.append(ConvSnp_norm(in_channels, out_channels, activation))
    for _ in range(n_block - 1):
         layers.append(ConvSnp_norm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvSnp_DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_block,  activation='ReLU'):
        super(ConvSnp_DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConvSnp(in_channels, out_channels, n_block,  activation)
    def forward(self,x):
        out = self.maxpool(x)
        return self.nConvs(out)


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(decoder_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.activation = get_activation(activation)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = x

        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv3(out)

        x1 = self.bn2(x1)
        x1 = self.activation(x1)
        x1 = self.conv1(x1)

        out = torch.add(x1, out)
        return out

def make_decoder(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layer = []
    layer.append(decoder_block(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layer.append(decoder_block(out_channels, out_channels, activation))
    return nn.Sequential(*layer)

class ConvSnp_UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_block, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        #self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, (2, 2), 2)
        #self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #self.up = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)


        self.nConvs = make_decoder(in_channels, out_channels, n_block, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        x = torch.cat([skip_x, up], dim=1)
        return self.nConvs(x)



class DSP_UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_block, activation='ReLU'):
        super().__init__()
        self.nConvs = make_decoder(in_channels, out_channels, n_block, activation)

    def forward(self, x, skip_x):
        # up = self.up(x)
        out = torch.cat([skip_x, x], dim=1)
        return self.nConvs(out)

class DSP(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(DSP, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveMaxPool2d(bin),
                nn.ELU(),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(reduction_dim),
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='nearest'))
        return torch.cat(out, 1)



class SPC(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bins = (7, 14)
        in_channels = 64

        self.inc = _make_nConv(n_channels, in_channels, n_block=2)
        self.down1 = ConvSnp_DownBlock(in_channels, in_channels * 2, n_block=1)
        self.down2 = ConvSnp_DownBlock(in_channels * 2, in_channels * 4, n_block=1)
        self.down3 = ConvSnp_DownBlock(in_channels * 4, in_channels * 8, n_block=1)

        self.dsp = DSP(in_channels * 8, in_channels * 2, bins)

        self.up4 = DSP_UpBlock(in_channels * 16+in_channels * 4, in_channels * 4, n_block=2)
        self.up3 = ConvSnp_UpBlock(in_channels * 8, in_channels * 2, n_block=2)
        self.up2 = ConvSnp_UpBlock(in_channels * 4, in_channels, n_block=2)
        self.up1 = ConvSnp_UpBlock(in_channels * 2, in_channels, n_block=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.dsp(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)  # if nusing BCEWithLogitsLoss or class>1
        return logits