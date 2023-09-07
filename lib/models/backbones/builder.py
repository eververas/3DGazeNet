from .resnet import BasicBlock, Bottleneck, ResNet
from .mobile_vit import MobileVit
from .neck import ResNeck, MobileNeck
from lib.utils.defaults import BACKBONES
import torch.nn as nn

mobilevit_spec = {'small': ([64, 80, 96],
                            [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 96]),
                  'medium': ([96, 120, 144],
                             [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]),
                  'large': ([144, 192, 240],
                            [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640])}

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}
dim_in = 9


def get_mobilevit_backbone(cfg):
    size = cfg.MODEL.MOBILE_VIT_SIZE
    dims, channels = mobilevit_spec[size]
    backbone = MobileVit(dims_in=dim_in, dims=dims, channels=channels)
    return backbone, dim_in


def get_resnet_backbone(cfg):
    if cfg.MODEL.NUM_LAYERS == 9:
        backbone = ResNet9(dim_in, 32, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9)
    else:
        block_class, layers = resnet_spec[cfg.MODEL.NUM_LAYERS]
        backbone = ResNet(block_class, layers, cfg.MODEL, dim_in=dim_in)
    return backbone, dim_in


def get_mobilevit_neck(cfg):
    size = cfg.MODEL.MOBILE_VIT_SIZE
    _, channels = mobilevit_spec[size]
    in_features, out_features = channels[-2], channels[-1]
    return MobileNeck(in_features, out_features)


def get_resnet_neck(cfg):
    num_layers = cfg.MODEL.NUM_LAYERS
    img_size = cfg.MODEL.IMAGE_SIZE
    stride = 2
    if num_layers in [9]:
        in_features = 128
        out_features = 64
        stride = 4
    elif num_layers in [18, 34]:
        in_features = 512
        out_features = 512
    elif num_layers in [50, 101, 152]:
        in_features = 2048
        out_features = 512
    else:
        raise "Problem with input output features of the neck"
    return ResNeck(in_features, out_features, stride=stride, bias=False)


def build_backbone(cfg):
    """Build backbone"""
    type = cfg.MODEL.BACKBONE_TYPE.lower()
    assert type in BACKBONES, "Backbone type is not supported"
    get_backbone_func = f'get_{type}_backbone(cfg)'
    return eval(get_backbone_func)


def build_neck(cfg, type):
    assert type.lower() in BACKBONES, "Unable to defined neck for the backbone due to non existent name "
    get_neck_func = f'get_{type.lower()}_neck(cfg)'
    return eval(get_neck_func)
