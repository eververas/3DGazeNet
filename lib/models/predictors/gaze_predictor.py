import torch
import torch.nn as nn

from ..backbones.builder import build_backbone
from ..backbones.builder import build_neck
from .utils import cls_init_weights

from lib import HOOKS


@HOOKS.register_module('gaze_predictor')
class GazePredictor(nn.Module):
    def __init__(self, cfg):
        super(GazePredictor, self).__init__()
        img_size = cfg.MODEL.IMAGE_SIZE[0]
        self.num_points_out_face = cfg.MODEL.NUM_POINTS_OUT_FACE

        self.encoder, dim_in = build_backbone(cfg)
        self.encoder.init_weights(init_func=cls_init_weights, pretrained=cfg.MODEL.BACKBONE_PRETRAINED)

        self.neck = build_neck(cfg, type(self.encoder).__name__)
        self.neck.init_weights(cls_init_weights)
        
        with torch.no_grad():
            nz_feat = self.neck(self.encoder(torch.rand(1, dim_in, img_size, img_size))[0])[0].shape[1]
        self.pred_layer_gaze = nn.Linear(nz_feat, 3)
        self.pred_layer_points_face = nn.Linear(nz_feat, self.num_points_out_face * 3)

        self.apply(cls_init_weights)

    def forward(self, x):
        batch_size = x.shape[0]

        feat = self.encoder(x)[0]
        reduced_features_gaze, reduced_features_face = self.neck(feat)
        # predict gaze
        gaze = self.pred_layer_gaze(reduced_features_gaze).view(batch_size, 3)
        gaze = gaze / torch.norm(gaze, dim=1, keepdim=True)
        # predict face verts
        verts_face = self.pred_layer_points_face(reduced_features_face).view(batch_size, self.num_points_out_face, 3)

        return gaze, verts_face
