import torch.nn as nn
import torch


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x



def conv_nxn_bn(inp, oup, kernel_size=3, stride=1, padding=1, use_act=True, bias=False,use_bias=False):
    the_module = nn.Sequential()
    the_module.add_module(name='conv2d', module=nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias))
    if use_bias:
        the_module.add_module(name='bn', module=nn.BatchNorm2d(oup))
    if use_act:
        the_module.add_module(name=f'silu', module=SiLU())
    return the_module


class BaseNeck(nn.Module):
    def __init__(self):
        super(BaseNeck, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        reduce_gaze = self.reduce_layer_gaze(x).view(batch_size, -1)
        reduce_face = self.reduce_layer_face(x).view(batch_size, -1)
        return reduce_gaze, reduce_face

    def init_weights(self, init_func):
        self.apply(init_func)


class ResNeck(BaseNeck):
    def __init__(self, in_features, out_features, kernel=3, stride=2, padding=1, bias=True):
        super(ResNeck, self).__init__()
        self.reduce_layer_gaze = nn.Conv2d(in_features, out_features, kernel, stride, padding, bias=bias)

        self.reduce_layer_face = nn.Conv2d(in_features, out_features, kernel, stride, padding, bias=bias)


class MobileVitNeck(BaseNeck):
    def __init__(self, in_features, out_features):
        super(MobileVitNeck, self).__init__()
        self.reduce_layer_gaze = conv_nxn_bn(in_features, out_features, 1, 1, 0, bias=False)
        self.reduce_layer_face = conv_nxn_bn(in_features, out_features, 1, 1, 0, bias=False)
