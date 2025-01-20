import os
import torch
import torch.nn as nn
from collections import OrderedDict

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 doResidual=True, bounded=False, expansion=None):
        super(BasicBlock, self).__init__()
        self.expansion = BasicBlock.get_expansion(expansion)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU6(inplace=True) if bounded else nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.doResidual = doResidual
        self.bounded = bounded
        self.conv_bn_pairs = [(self.conv1, self.bn1), (self.conv2, self.bn2)]

    @classmethod
    def get_expansion(cls, expansion):
        del expansion  # unused
        return 1

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.bounded:
            out = torch.clamp(out, -6.0, 6.0)

        if self.doResidual:
            out += residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 bounded=False, expansion=None):
        super(Bottleneck, self).__init__()
        self.expansion = Bottleneck.get_expansion(expansion)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU6(inplace=True) if bounded else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.bounded = bounded
        self.conv_bn_pairs = [
            (self.conv1, self.bn1),
            (self.conv2, self.bn2),
            (self.conv3, self.bn3),
        ]

    @classmethod
    def get_expansion(cls, expansion):
        return expansion or 4

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.bounded:
            out = torch.clamp(out, -6.0, 6.0)

        return self.relu(out + residual)


class ResNet(nn.Module):

    def __init__(self, block, layers, mcfg=None, dim_in=3, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.bounded = False
        self.relu = nn.ReLU(inplace=True)
        self.expansion = block.get_expansion(None)

        strides = [1, 2, 2, 2]
        block_filters = [64, 128, 256, 512]

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, 128, kernel_size=3, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv_bn_pairs = [(self.conv1[-1], self.bn1)]

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_filters[0], layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, block_filters[1], layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, block_filters[2], layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, block_filters[3], layers[3], stride=strides[3])

        # self.init_weights(pretrained=mcfg.BACKBONE_PRETRAINED)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample_modules = [
                nn.Conv2d(self.inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM),
            ]
            if self.bounded:
                downsample_modules.append(
                    nn.Hardtanh(min_val=-6.0, max_val=6.0, inplace=True))
            downsample = nn.Sequential(*downsample_modules)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            bounded=self.bounded, expansion=self.expansion))
        self.inplanes = planes * self.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bounded=self.bounded,
                                expansion=self.expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x1m = self.maxpool(x1)
        x2 = self.layer1(x1m)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x5, (x1, x2, x3, x4, x5)
