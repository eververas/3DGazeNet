import torch.nn as nn
from lib.models.backbones.utils import conv_1x1_bn, conv_nxn_bn


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


class MobileNeck(BaseNeck):
    def __init__(self, in_features, out_features):
        super(MobileNeck, self).__init__()
        self.reduce_layer_gaze = conv_1x1_bn(in_features, out_features)
        self.reduce_layer_face = conv_1x1_bn(in_features, out_features)


if __name__ == '__main__':
    res = ResNeck(512, 256, stride=1)
    mob = MobileNeck(512, 512)
    print()
