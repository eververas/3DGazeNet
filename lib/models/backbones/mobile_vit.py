import torch
import torch.nn as nn
from lib.models.backbones.utils import SiLU, LayerNorm, conv_nxn_bn, conv_1x1_bn

from collections import OrderedDict

import os.path as osp
from lib.utils.utils import get_logger

logger = get_logger(__name__)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        # self.norm = nn.BatchNorm2d(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.fn(self.norm(x), **kwargs)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.dp = nn.Dropout(dropout)
        self.ac = SiLU()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.fc1(x.reshape((-1, self.dim)))
        x = self.ac(self.dp(x))
        x = self.dp(self.fc2(x)).reshape((b, c, h, w))
        return x


def get_t(x, h):
    b, p, n, hd = x.shape
    x = x.reshape((b * p, n, h, hd // h)).permute(0, 2, 1, 3)
    return x


class Hidim_fc(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim, dim_out)
        self.dim = dim
        self.dim_out = dim_out

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.fc(x.reshape((-1, self.dim)))
        x = x.reshape((b, c, h, self.dim_out))
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = Hidim_fc(dim, inner_dim * 3)  # nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            Hidim_fc(inner_dim, dim),  # nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.dim = dim

    def forward(self, x):
        qkv = self.to_qkv(x)
        # qkv = qkv.split(32,3)
        qkv = qkv.chunk(3, dim=-1)

        q, k, v = map(lambda t: get_t(t, h=self.heads), qkv)  # 'b p n (h d) -> b p h n d',
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        b, h, n, d = out.shape
        out = out.permute(0, 2, 1, 3).reshape((b // 4, 4, n, h * d))
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 1, 32, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        b, c, h, w = x.shape
        x = x.reshape((b * c, h // 2, 2, w)).permute(0, 1, 3, 2).reshape((b * c, h // 2 * w // 2, 2, 2)).permute(0, 1,
                                                                                                                 3,
                                                                                                                 2).reshape(
            (b, c, h * w // 4, 4)).permute(0, 3, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 3, 2, 1).reshape((b * c, 4, h // 2, w // 2)).permute(0, 2, 1, 3).reshape(
            (b * c, h, 2, w // 2)).permute(0, 1, 3, 2).reshape((b, c, h, w))

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileVit(nn.Module):
    def __init__(self, dims_in, dims, channels, expansion=4, kernel_size=3, patch_size=(2, 2), stacking_outputs=False):
        super().__init__()
        self.stacking = stacking_outputs

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(dims_in, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        # # 1x80x4x4
        # self.pred_layer_gaze = nn.Linear(1536, 3, bias=False)
        # self.pred_layer_points_face = nn.Linear(1536, 68 * 3, bias=False)

    def forward(self, x):
        output = []

        # stage 0
        x = self.conv1(x)
        x = self.mv2[0](x)

        # stage 1
        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)  # Repeat
        output.append(x)

        # stage 2
        x = self.mv2[4](x)
        x = self.mvit[0](x)
        output.append(x)

        # stage 3
        x = self.mv2[5](x)
        x = self.mvit[1](x)
        output.append(x)

        # stage 4
        x = self.mv2[6](x)
        x = self.mvit[2](x)
        output.append(x)

        return x, tuple(output)

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
            self.load_state_dict(state_dict, strict=True)
        else:
            self.apply(init_func)


def mobilevit_small(dims_in):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 96]
    return MobileVit(dims_in=dims_in, dims=dims, channels=channels, expansion=4), channels


def mobilevit_medium(dims_in):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileVit(dims_in=dims_in, dims=dims, channels=channels), channels


def mobilevit_large(dims_in):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileVit(dims_in=dims_in, dims=dims, channels=channels), channels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from thop import profile

    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 96]
    net = MobileVit(dims_in=9, dims=dims, channels=channels, expansion=4)

    x = torch.rand(1, 9, 256, 256)

    macs, params = profile(net, inputs=(x,))
    print("Num params: %.3f M" % (params / 1e6))
    # print("num train params: ", sum(p.numel() for p in net.parameters() if p.requires_grad))
    print("Num ops: %.3f GFlops" % (2. * macs / 1e9))
