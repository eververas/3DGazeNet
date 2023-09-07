from .loss_components import *
import torch.nn as nn
from lib import HOOKS
from lib.utils import load_eyes3d


@HOOKS.register_module('base_loss')
class BaseLoss(nn.Module):
    def __init__(self, cfg, mode):
        super(BaseLoss, self).__init__()
        self.mode = mode
        self.img_size = int(cfg.MODEL.IMAGE_SIZE[0])

        # loss weights
        self.w_vertex_eyes_reg = cfg.LOSS.VERTEX_REGRESSION_WEIGHT_EYES
        self.w_vertex_face_reg = cfg.LOSS.VERTEX_REGRESSION_WEIGHT_FACE
        self.w_edge_length_eyes = cfg.LOSS.EDGE_LENGTH_LOSS_WEIGHT_EYES
        self.w_edge_length_face = cfg.LOSS.EDGE_LENGTH_LOSS_WEIGHT_FACE
        self.w_gaze_acos = cfg.LOSS.GAZE_ACOS_WEIGHT

        # loss names
        if self.mode == 'vertex':
            self.loss_names = ['L_l1', 'L_edge_length', 'L_gaze', 'L_gaze_vector', 'L_gaze_face', 'L_gaze_comb']
            self.loss_names_opt = ['L_l1', 'L_edge_length', 'L_gaze', 'L_gaze_vector']
            self.loss_name_performance = 'L_gaze_comb'
        else:
            self.loss_names = ['L_gaze_vector', 'L_gaze_pitchyaws']
            self.loss_names_opt = ['L_gaze_vector']
            self.loss_name_performance = 'L_gaze_vector'

        # 3d eyeball data
        eyes3d_dict = load_eyes3d('data/eyes3d.pkl')
        iris_idxs481 = eyes3d_dict['iris_idxs481']
        trilist_eye = eyes3d_dict['trilist_eye']
        eye_template_homo = eyes3d_dict['eye_template_homo']
        self.trilist_eye = torch.tensor(trilist_eye.astype(int))
        self.iris_lms_idx = torch.tensor(iris_idxs481.astype(int))
        self.mean_l = torch.tensor(eye_template_homo['left'][:, :3].astype(np.float32))
        self.mean_r = torch.tensor(eye_template_homo['right'][:, :3].astype(np.float32))

    def forward_vertex(self, output, meta):

        unboxed = unbox_output_eyes(output, meta, self.img_size)
        verts_pred_l, verts_pred_r, verts_trg_l, verts_trg_r, gaze_pred = unboxed

        # gaze vectors from eyes pred + trg
        gaze_vec_pred_l = points_to_vector(verts_pred_l * (-1), self.iris_lms_idx)
        gaze_vec_pred_r = points_to_vector(verts_pred_r * (-1), self.iris_lms_idx)
        gaze_vec_trg_l  = points_to_vector(verts_trg_l * (-1), self.iris_lms_idx)
        gaze_vec_trg_r  = points_to_vector(verts_trg_r * (-1), self.iris_lms_idx)
        gaze_vec_pred_f   = gaze_vec_pred_r + gaze_vec_pred_l
        gaze_vec_pred_f   = gaze_vec_pred_f / torch.norm(gaze_vec_pred_f, dim=1, keepdim=True)
        gaze_vec_trg_f    = gaze_vec_trg_r + gaze_vec_trg_l
        gaze_vec_trg_f    = gaze_vec_trg_f / torch.norm(gaze_vec_trg_f, dim=1, keepdim=True)
        gaze_vec_combined = gaze_vec_pred_f + gaze_pred['vector']
        gaze_vec_combined = gaze_vec_combined / torch.norm(gaze_vec_combined, dim=1, keepdim=True)

        # left ----------------------------------------------------------------------------------
        # reconstruction loss
        l1_loss_l, lx_l, ly_l, lz_l = l1_recon_loss(
                    self.w_vertex_eyes_reg, verts_pred_l, verts_trg_l)
        # edge loss
        edge_length_loss_l = edge_length_loss(
                    self.trilist_eye, self.w_edge_length_eyes, verts_pred_l, verts_trg_l)
        # gaze loss
        gaze_loss_l, gla_l = gaze_loss_from_vectors(
                    self.w_gaze_acos, gaze_vec_pred_l, gaze_vec_trg_l)

        # right ----------------------------------------------------------------------------------
        # reconstruction loss
        l1_loss_r, lx_r, ly_r, lz_r = l1_recon_loss(
                    self.w_vertex_eyes_reg, verts_pred_r, verts_trg_r)
        # edge loss
        edge_length_loss_r = edge_length_loss(
                    self.trilist_eye, self.w_edge_length_eyes, verts_pred_r, verts_trg_r)
        # gaze loss
        gaze_loss_r, gla_r = gaze_loss_from_vectors(
                    self.w_gaze_acos, gaze_vec_pred_r, gaze_vec_trg_r)

        # combine ----------------------------------------------------------------------------------
        l1_loss = l1_loss_l + l1_loss_r
        edge_length_loss = edge_length_loss_l + edge_length_loss_r
        gaze_loss = gaze_loss_l + gaze_loss_r
        # vector gaze loss
        gaze_loss_vec, _ = gaze_loss_from_vectors(
                    self.w_gaze_acos, gaze_pred['vector'], gaze_vec_trg_f)
        # face gaze loss
        gaze_loss_face, _ = gaze_loss_from_vectors(
                    self.w_gaze_acos, gaze_vec_pred_f, gaze_vec_trg_f)
        # points+vector loss
        gaze_loss_comb, gaze_loss_comb_array = gaze_loss_from_vectors(
                    self.w_gaze_acos, gaze_vec_combined, gaze_vec_trg_f)

        losses = [l1_loss, edge_length_loss, gaze_loss, gaze_loss_vec, gaze_loss_face, gaze_loss_comb]
        return {loss_name: loss for loss_name, loss in zip(self.loss_names, losses)}

    def forward_gaze(self, output, meta):
        # get gaze predictions and targets
        gaze_pred, _, _, gaze_trg_f = unbox_output_gaze(output, meta)
        # gaze loss
        gaze_vector_loss, gaze_loss_comb_array = gaze_loss_from_vectors(
                    self.w_gaze_acos, gaze_pred['vector'], gaze_trg_f['vector'])
        gaze_pitchyaws_loss = gaze_loss_from_pitchyaws(
                    self.w_gaze_acos, gaze_pred['pitchyaws'], gaze_trg_f['pitchyaws'])

        losses = [gaze_vector_loss, gaze_pitchyaws_loss]
        return {loss_name: loss for loss_name, loss in zip(self.loss_names, losses)}

    def forward(self, output, meta):
        return eval(f'self.forward_{self.mode}(output, meta)')
