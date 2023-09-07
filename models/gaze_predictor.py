from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from .builder import build_backbone
from .builder import build_neck
from utils import get_input_and_transform, show_result, Timer, points_to_vector, load_eyes3d, trans_verts_from_patch_to_org


class GazePredictorHandler:
    def __init__(self, cfg, device='cuda:0'):
        self.predict_eyes = True if cfg.MODE == 'vertex' else False
        self.model = GazePredictor(cfg=cfg, predict_eyes=self.predict_eyes).to(device)
        self.device = device
        pretrained_ckpt = torch.load(cfg.PRETRAINED, map_location=lambda storage, loc: storage)
        if isinstance(pretrained_ckpt, OrderedDict):
            state_dict = pretrained_ckpt
        elif isinstance(pretrained_ckpt, dict) and 'state_dict' in pretrained_ckpt:
            state_dict = pretrained_ckpt['state_dict']
        else:
            raise "Unable to recognize state dict"
        eyes3d_dict = load_eyes3d()
        self.iris_idxs481 = eyes3d_dict['iris_idxs481']
        self.trilist_eye = eyes3d_dict['trilist_eye']
        self.eye_template_homo = eyes3d_dict['eye_template_homo']
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.crop_width, self.crop_height = cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1]
        self.mean = torch.as_tensor([0.485, 0.456, 0.406])
        self.std = torch.as_tensor([0.229, 0.224, 0.225])
        self.face_elements = ['left_eye', 'right_eye', 'face']

    @Timer(name='GazePredictor', fps=True, pprint=False)
    def __call__(self, img, lms5, *args, **kwargs):
        out_dict = {}
        input_size = self.crop_width
        input_args = (img, [self.crop_width, self.crop_height], 0, False)
        diag1 = np.linalg.norm(lms5[0] - lms5[4])
        diag2 = np.linalg.norm(lms5[1] - lms5[3])
        diag = np.max([diag1, diag2])
        # face_crop_len = int(4 * diag)
        face_crop_len = int(1.5 * diag)
        eyes_crop_len = int(2 * diag / 5)
        centers = [lms5[1], lms5[0], lms5[2]]
        crop_info = {'left_eye' : {'center': centers[0], 'crop_len': [eyes_crop_len, eyes_crop_len]},
                     'right_eye': {'center': centers[1], 'crop_len': [eyes_crop_len, eyes_crop_len]},
                     'face'     : {'center': centers[2], 'crop_len': [face_crop_len, face_crop_len]}}

        input_list = []
        trans_list = []
        for eye_str in self.face_elements:
            cnt = crop_info[eye_str]['center']
            crop_len = crop_info[eye_str]['crop_len']
            # resize to model preferences
            trans, img_patch_cv = get_input_and_transform(cnt, crop_len, *input_args)
            img_patch_cv = np.transpose(img_patch_cv, (2, 0, 1)) / 255  # (C,H,W) and between 0,1
            img_patch_torch = torch.as_tensor(img_patch_cv, dtype=torch.float32)  # to torch and from int to float
            img_patch_torch.sub_(self.mean.view(-1, 1, 1)).div_(self.std.view(-1, 1, 1))
            input_list += [img_patch_torch]
            trans_list += [trans]
        # invert transforms image -> patch
        trans_list_inv = []
        for i in range(2):
            try:
                trans_list[i] = np.concatenate((trans_list[i], np.array([[0, 0, 1]])))
                trans_list_inv += [np.linalg.inv(trans_list[i])[:2]]
            except:
                print(f'Error inverting bbox crop transform')
                return None

        # Model Inference ---------------------------------------------------------------------------
        model_input = torch.cat(input_list).unsqueeze(0).to(self.device)
        verts_eyes, verts_face, gaze = self.model(model_input)

        if self.predict_eyes:
            verts_left = verts_eyes[:, :int(verts_eyes.shape[1] / 2)].clone()
            verts_right = verts_eyes[:, int(verts_eyes.shape[1] / 2):].clone()
            # calculate gaze from eyes
            gaze_left = points_to_vector(verts_left * (-1), self.iris_idxs481)
            gaze_right = points_to_vector(verts_right * (-1), self.iris_idxs481)
            gaze_face = gaze_left + gaze_right
            gaze_face /= torch.norm(gaze_face, dim=1, keepdim=True) 
            # calculate combined gaze
            gaze_combined = gaze_face + gaze
            gaze_combined /= torch.norm(gaze_combined, dim=1, keepdim=True)
            # move to cpu
            verts_left = verts_left.detach().cpu().numpy()
            verts_right = verts_right.detach().cpu().numpy()
            gaze_left = gaze_left.detach().cpu().numpy()
            gaze_right = gaze_right.detach().cpu().numpy()
            gaze_face = gaze_face.detach().cpu().numpy()
            gaze_combined = gaze_combined.detach().cpu().numpy()
            # undo scale+translation patch -> pred_space
            verts_left[:, :, [0, 1]] = (verts_left[:, :, [0, 1]] + 1.) * (input_size / 2)
            verts_left[:, :, 2] *= (input_size / 2)
            verts_right[:, :, [0, 1]] = (verts_right[:, :, [0, 1]] + 1.) * (input_size / 2)
            verts_right[:, :, 2] *= (input_size / 2)
            # undo transform image -> patch
            verts_left_in_img = trans_verts_from_patch_to_org(
                verts_left[0], eyes_crop_len, input_size, trans=trans_list_inv[0])
            verts_right_in_img = trans_verts_from_patch_to_org(
                verts_right[0], eyes_crop_len, input_size, trans=trans_list_inv[1])
        gaze = gaze.detach().cpu().numpy()

        # make output dict
        out_dict.update({
            'gaze': gaze[0],
            'centers': {
                'left': centers[0].astype(np.int),
                'right': centers[1].astype(np.int),
                'face': centers[2].astype(np.int)},
            'iris_idxs': None,
            'verts_eyes': None,
            'centers_iris': None,
            'gaze_from_eyes': None,
            'gaze_combined': None
        })
        if self.predict_eyes:
            out_dict['verts_eyes'] = {'left': verts_left_in_img, 'right': verts_right_in_img}
            out_dict['centers_iris'] = {
                'left': verts_left_in_img[self.iris_idxs481][:, :2].mean(axis=0),
                'right': verts_right_in_img[self.iris_idxs481][:, :2].mean(axis=0)}
            out_dict['iris_idxs'] = self.iris_idxs481
            out_dict['gaze_from_eyes'] = {'left':  gaze_left[0], 'right': gaze_right[0], 'face':  gaze_face[0]}
            out_dict['gaze_combined'] = gaze_combined[0]

        return out_dict


class GazePredictor(nn.Module):

    def __init__(self, cfg, predict_eyes=True):
        super(GazePredictor, self).__init__()
        img_size = cfg.IMAGE_SIZE[0]
        self.num_points_out_face = cfg.NUM_POINTS_OUT_FACE
        self.num_points_out_eyes = cfg.NUM_POINTS_OUT_EYES
        self.num_points_out_gaze = 3  # 3 for verctor, 2 for pitchyaws
        self.predict_eyes = predict_eyes
        
        self.encoder, dim_in = build_backbone(cfg)
        self.neck = build_neck(cfg, type(self.encoder).__name__)

        with torch.no_grad():
            nz_feat = self.neck(self.encoder(torch.rand(1, dim_in, img_size, img_size))[0])[0].shape[1]
        if self.predict_eyes:
            self.pred_layer_points_eyes = nn.Linear(nz_feat, self.num_points_out_eyes * 3)
            self.pred_layer_gaze_vec = nn.Linear(nz_feat, self.num_points_out_gaze)
        else:
            self.pred_layer_gaze = nn.Linear(nz_feat, self.num_points_out_gaze)
        self.pred_layer_points_face = nn.Linear(nz_feat, self.num_points_out_face * 3)

    @Timer(name='Forward Gaze Predictor', fps=True, pprint=False)
    def forward(self, x):
        batch_size = x.shape[0]

        feat = self.encoder(x)[0]
        reduced_features_eyes, reduced_features_face = self.neck(feat)
        
        verts_eyes = None
        if self.predict_eyes:
            verts_eyes = self.pred_layer_points_eyes(reduced_features_eyes).view(batch_size, self.num_points_out_eyes, 3)
            vecs_gaze = self.pred_layer_gaze_vec(reduced_features_eyes).view(batch_size, self.num_points_out_gaze)
            vecs_gaze = vecs_gaze / torch.norm(vecs_gaze, dim=1, keepdim=True)
        else:
            vecs_gaze = self.pred_layer_gaze(reduced_features_eyes).view(batch_size, self.num_points_out_gaze)
            vecs_gaze = vecs_gaze / torch.norm(vecs_gaze, dim=1, keepdim=True)
        verts_face = self.pred_layer_points_face(reduced_features_face).view(batch_size, self.num_points_out_face, 3)

        return verts_eyes, verts_face, vecs_gaze
