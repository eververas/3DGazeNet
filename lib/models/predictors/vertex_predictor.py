import torch
import torch.nn as nn

from ..backbones.builder import build_backbone
from ..backbones.builder import build_neck

from .utils import cls_init_weights
from lib import HOOKS


@HOOKS.register_module('vertex_predictor')
class VertexPredictor(nn.Module):
    def __init__(self, cfg):
        super(VertexPredictor, self).__init__()
        img_size = cfg.MODEL.IMAGE_SIZE[0]
        self.num_points_out_eyes = cfg.MODEL.NUM_POINTS_OUT_EYES
        self.num_points_out_face = cfg.MODEL.NUM_POINTS_OUT_FACE

        self.encoder, dim_in = build_backbone(cfg)
        self.encoder.init_weights(init_func=cls_init_weights, pretrained=cfg.MODEL.BACKBONE_PRETRAINED)

        self.neck = build_neck(cfg, type(self.encoder).__name__)
        self.neck.init_weights(cls_init_weights)

        with torch.no_grad():
            nz_feat = self.neck(self.encoder(torch.rand(1, dim_in, img_size, img_size))[0])[0].shape[1]
        self.pred_layer_points_eyes = nn.Linear(nz_feat, self.num_points_out_eyes * 3)
        self.pred_layer_points_face = nn.Linear(nz_feat, self.num_points_out_face * 3)
        self.pred_layer_gaze_vec = nn.Linear(nz_feat, 3)

        self.apply(cls_init_weights)

    # def forward(self,x, mode='vertex'):
    #     return eval(f'self.forward_{mode}(x)')

    def forward(self, x):
        batch_size = x.shape[0]

        feat = self.encoder(x)[0]
        reduced_features_eyes, reduced_features_face = self.neck(feat)
        # predict eye+gaze
        verts_eyes = self.pred_layer_points_eyes(reduced_features_eyes).view(batch_size, self.num_points_out_eyes, 3)
        vecs_gaze = self.pred_layer_gaze_vec(reduced_features_eyes).view(batch_size, 3)
        vecs_gaze = vecs_gaze / torch.norm(vecs_gaze, dim=1, keepdim=True)
        # predict face
        verts_face = self.pred_layer_points_face(reduced_features_face).view(batch_size, self.num_points_out_face, 3)

        return verts_eyes, verts_face, vecs_gaze
