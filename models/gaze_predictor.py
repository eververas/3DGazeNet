from collections import OrderedDict
import _pickle as cPickle

import torch
import torch.nn as nn
import numpy as np

from .builder import build_backbone
from .builder import build_neck
from utils import get_input_and_transform, show_result, Timer, get_gaze_pitchyaws_from_vectors, \
    get_gaze_pitchyaws_from_eyes


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
        iris_data = self.load_eyes3d()
        self.iris_idxs, self.idxs481, self.iris_idxs481, self.idxs288, self.iris_idxs288, self.trilist_eye, \
        self.eye_template_homo = iris_data
        self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()
        self.crop_width, self.crop_height = cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1]
        # pixel transformation for cnn model
        self.mean = torch.as_tensor([0.485, 0.456, 0.406])
        self.std = torch.as_tensor([0.229, 0.224, 0.225])
        self.face_elements = ['left_eye', 'right_eye', 'face']

    @Timer(name='GazePredictor', fps=True, pprint=False)
    def __call__(self, img, lms5, *args, **kwargs):
        out_eyes = {}
        part_to_input_args = (img, [self.crop_width, self.crop_height], 0, False)
        diag1 = np.linalg.norm(lms5[0] - lms5[4])
        diag2 = np.linalg.norm(lms5[1] - lms5[3])
        diag = np.max([diag1, diag2])
        face_crop_len = int(4 * diag)
        eyes_crop_len = int(2 * diag / 5)

        centers = [lms5[1], lms5[0], lms5[2]]
        crop_info = {'left_eye': {'center': centers[0], 'crop_len': [eyes_crop_len, eyes_crop_len]},
                     'right_eye': {'center': centers[1], 'crop_len': [eyes_crop_len, eyes_crop_len]},
                     'face': {'center': centers[2], 'crop_len': [face_crop_len, face_crop_len]}}

        input_list = []
        # scale = []
        # trsl = []
        for eye_str in self.face_elements:
            cnt = crop_info[eye_str]['center']
            crop_len = crop_info[eye_str]['crop_len']
            # resize to model preferences
            trans, img_patch_cv = get_input_and_transform(cnt, crop_len, *part_to_input_args)
            np_img_patch_copy = img_patch_cv.copy()
            np_img_patch_copy = np.transpose(np_img_patch_copy, (2, 0, 1)) / 255  # (C,H,W) and between 0,1
            torch_img_patch = torch.as_tensor(np_img_patch_copy, dtype=torch.float32)  # torch and from int to float
            torch_img_patch.sub_(self.mean.view(-1, 1, 1)).div_(self.std.view(-1, 1, 1))
            input_list.append(torch_img_patch)

        # Model Inference ---------------------------------------------------------------------------
        model_input = torch.cat(input_list).unsqueeze(0).to(self.device)
        verts_eyes, verts_face, gaze = self.model(model_input)

        gaze_normalized_vector = gaze.detach().cpu().numpy()

        if self.predict_eyes:
            gaze_normalized_vector[:, 2] *= -1
            verts_eyes = verts_eyes.detach().cpu().numpy().reshape(2, verts_eyes.shape[1] // 2, 3)

        # put points back on original image -----------------------------------------------------------------------
        for ei in range(2):
            eye_str = self.face_elements[ei]
            # transform output to original image space
            gaze_angle_from_vector = get_gaze_pitchyaws_from_vectors(gaze_normalized_vector)
            out_eyes[eye_str] = {'gaze_angle_from_vector': gaze_angle_from_vector,
                                 'eye_center': centers[ei].astype(np.int)}
            if self.predict_eyes:
                gaze_angle_from_eye, eyes_normalized_vector = get_gaze_pitchyaws_from_eyes(verts_eyes[ei],
                                                                                           self.iris_idxs)
                out_eyes[eye_str]['gaze_angle_from_eyes'] = gaze_angle_from_eye

        # get gaze from points
        # angles = (np.array(angles_l) + np.array(angles_r)) / 2.
        # # fix scaling so the size of the eyeballs match
        # out_l = out_eyes['left_eye']
        # out_r = out_eyes['right_eye']
        #
        # radious_l = np.linalg.norm(out_l[0] - out_l[:32].mean(axis=0))
        # radious_r = np.linalg.norm(out_r[0] - out_r[:32].mean(axis=0))
        # scl_l = 1. + (1. - radious_l / radious_r) / 2
        # scl_r = 1. + (1. - radious_r / radious_l) / 2
        # cnt_l = out_l.mean(axis=0)
        # cnt_r = out_r.mean(axis=0)
        # out_l = scl_l * (out_l - cnt_l) + cnt_l
        # out_r = scl_r * (out_r - cnt_r) + cnt_r

        # # # get mean prediction using both eyes --------------------------------------------------------------------
        # iris8_l = out_eyes['left_eye'][-8:]
        # iris8_r = out_eyes['right_eye'][-8:]
        # #
        # # # translate to predicted center of iris
        # out_l -= out_l[self.idxs_iris].mean(axis=0) - iris8_l.mean(axis=0)
        # out_r -= out_r[self.idxs_iris].mean(axis=0) - iris8_r.mean(axis=0)
        # #
        # # # put results on output dict
        # out_eyes['left_eye'] = out_l
        # out_eyes['right_eye'] = out_r
        # out_eyes['angles'] = angles
        return out_eyes

    @staticmethod
    def load_eyes3d(fname_eyes3d='data/eyes3d.pkl'):
        with open(fname_eyes3d, 'rb') as f:
            eyes3d = cPickle.load(f)
        iris_idxs = eyes3d['left_iris_lms_idx']
        idxs481 = eyes3d['mask481']['idxs']
        iris_idxs481 = eyes3d['mask481']['idxs_iris']
        idxs288 = eyes3d['mask288']['idxs']
        iris_idxs288 = eyes3d['mask288']['idxs_iris']
        trilist_eye = eyes3d['mask481']['trilist']
        eyel_template = eyes3d['left_points'][idxs481]
        eyer_template = eyes3d['right_points'][idxs481]
        eye_template = {
            'left_eye': eyes3d['left_points'][idxs481],
            'right_eye': eyes3d['right_points'][idxs481]
        }
        eye_template_homo = {
            'left_eye': np.append(eye_template['left_eye'], np.ones((eyel_template.shape[0], 1)), axis=1),
            'right_eye': np.append(eye_template['right_eye'], np.ones((eyer_template.shape[0], 1)), axis=1)
        }

        return iris_idxs, idxs481, iris_idxs481, idxs288, iris_idxs288, trilist_eye, eye_template_homo


class GazePredictor(nn.Module):

    def __init__(self, cfg, predict_eyes=True):
        super(GazePredictor, self).__init__()

        self.gaze_is_relative_to_face = cfg.GAZE_IS_RELATIVE_TO_FACE
        self.num_points_out_face = cfg.NUM_POINTS_OUT_FACE
        self.num_points_out_eyes = cfg.NUM_POINTS_OUT_EYES

        self.num_points_out_gaze = 3  # 3 for verctor, 2 for pitchyaws
        self.predict_eyes = predict_eyes
        img_size = cfg.IMAGE_SIZE[0]

        # dim_in = 9
        # block_class, layers = self.RESNET_SPEC[cfg.MODEL_LAYERS]
        # self.encoder = ResNet(block_class, layers, cfg, dim_in=dim_in)

        self.encoder, dim_in = build_backbone(cfg)
        # self.encoder.init_weights(init_func=cls_init_weights, pretrained=cfg.MODEL.BACKBONE_PRETRAINED)

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
            verts_eyes = self.pred_layer_points_eyes(reduced_features_eyes).view(batch_size, self.num_points_out_eyes,
                                                                                 3)
            vecs_gaze = self.pred_layer_gaze_vec(reduced_features_eyes).view(batch_size, 3)
            vecs_gaze = vecs_gaze / torch.norm(vecs_gaze, dim=1, keepdim=True)
        else:
            vecs_gaze = self.pred_layer_gaze(reduced_features_eyes).view(batch_size, 3)
            vecs_gaze = vecs_gaze / torch.norm(vecs_gaze, dim=1, keepdim=True)
        verts_face = self.pred_layer_points_face(reduced_features_face).view(batch_size, self.num_points_out_face, 3)

        return verts_eyes, verts_face, vecs_gaze
