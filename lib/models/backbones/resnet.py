from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import logging
from lib.models.backbones.utils import conv3x3
import torch.nn as nn
import torch
from collections import OrderedDict

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


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
    def get_expansion(cls, expansion=None):
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
        self.relu = nn.ReLU6(inplace=True) if self.bounded else nn.ReLU(inplace=True)

        self.expansion = block.get_expansion(None)

        num_blocks = len(layers)
        if mcfg is not None and mcfg.STRIDES is not None:
            if len(mcfg.STRIDES) != num_blocks:
                raise ValueError('STRIDES needs to have same length as layers')
            strides = mcfg.STRIDES
        else:
            strides = [1, 2, 2, 2]

        if mcfg is not None and mcfg.BLOCK_FILTERS is not None:
            if len(mcfg.BLOCK_FILTERS) != num_blocks:
                raise ValueError('BLOCK_FILTERS needs to have same length as layers')
            block_filters = mcfg.BLOCK_FILTERS
        else:
            block_filters = [64, 128, 256, 512]

        if dim_in == 3:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
            self.conv_bn_pairs = [(self.conv1, self.bn1)]
        else:
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

    def init_weights(self, init_func=None, pretrained=''):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
                :param init_func: Function to initialize modules
        """

        is_file = osp.isfile(pretrained)
        print(f'Pretrained file exists? :{is_file}')
        if is_file:
            logger.info('=> loading pretrained backbone {}'.format(pretrained))
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))

            if 'final_layer.weight' in state_dict.keys():
                logger.info('removing the final layers.')
                del state_dict['final_layer.weight']
                del state_dict['final_layer.bias']
                logger.info('=> loading pretrained backbone {}'.format(pretrained))
            self.apply(init_func)
            self.load_state_dict(state_dict, strict=False)
        else:
            self.apply(init_func)


if __name__ == '__main__':
    import torch
    from thop import profile
    from lib.utils.config import config

    resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
                   34: (BasicBlock, [3, 4, 6, 3]),
                   50: (Bottleneck, [3, 4, 6, 3]),
                   101: (Bottleneck, [3, 4, 23, 3]),
                   152: (Bottleneck, [3, 8, 36, 3])}

    block_class, layers = resnet_spec[152]
    backbone = ResNet(block_class, layers, config.MODEL, dim_in=9)
    x = torch.rand(1, 9, 128, 128)

    # print(net(x).shape)

    macs = profile(backbone, inputs=(x,))[0]
    print("Num params: %.3f M" % (sum(p.numel() for p in backbone.parameters()) / 1e6))
    # print("num train params: ", sum(p.numel() for p in net.parameters() if p.requires_grad))
    print("Num ops: %.3f GFlops" % (2. * macs / 1e9))
