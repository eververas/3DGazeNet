import torch.nn as nn
import torch
import math
import warnings


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            u = x.mean(3).unsqueeze(3)
            s = (x - u).pow(2).mean(3).unsqueeze(3)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, None, :] * x + self.bias[None, None, :]
            return x
        elif self.data_format == "channels_first":
            u = x.mean(1).unsqueeze(1)
            s = (x - u).pow(2).mean(1).unsqueeze(1)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1, padding=1, use_act=True, bias=False, use_batch_norm=True):
    the_module = nn.Sequential()
    the_module.add_module(name='conv2d', module=nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias))
    if use_batch_norm:
        the_module.add_module(name='bn', module=nn.BatchNorm2d(oup))
    if use_act:
        the_module.add_module(name=f'silu', module=SiLU())
    return the_module

