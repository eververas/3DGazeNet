import os
import copy
import tqdm
import numpy as np
import _pickle as cPickle

import torch

from ..utils import trans_coords_from_patch_to_org_3d, estimate_affine_matrix_3d23d, load_eyes3d, get_final_results_dir, points_to_vector


def inference(config, test_loader, model, device):
    # path_base = None
    model.eval()
    results = {}
    n_verts_eye = int(config.MODEL.NUM_POINTS_OUT_EYES / 2)

    eyes3d_dict = load_eyes3d('data/eyes3d.pkl')
    iris_idxs481 = eyes3d_dict['iris_idxs481']
    trilist_eye = eyes3d_dict['trilist_eye']
    eye_template = eyes3d_dict['eye_template']

    for i, (input_data, meta) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        # compute output
        output = model(input_data.to(device))
        # gather data to export
        batch_size = input_data.shape[0]
        input_size = config.MODEL.IMAGE_SIZE[0]
        if config.MODE == 'vertex':
            output_verts = output[0]
            output_vecs = output[-1]
            verts_pred_left = output_verts[:, :n_verts_eye].clone()
            verts_pred_right = output_verts[:, n_verts_eye:].clone()
            vecs_pred = output_vecs.clone()
            # calculate gaze from eyes
            vecs_pred_left = points_to_vector(verts_pred_left * (-1), iris_idxs481)
            vecs_pred_right = points_to_vector(verts_pred_right * (-1), iris_idxs481)
            vecs_pred_face = vecs_pred_left + vecs_pred_right
            vecs_pred_face /= torch.norm(vecs_pred_face, dim=1, keepdim=True) 
            # calculate combined gaze
            vecs_pred_combined = vecs_pred_face + vecs_pred
            vecs_pred_combined /= torch.norm(vecs_pred_combined, dim=1, keepdim=True)
            # move to cpu
            verts_pred_left = verts_pred_left.detach().cpu().numpy()
            verts_pred_right = verts_pred_right.detach().cpu().numpy()
            vecs_pred_left = vecs_pred_left.detach().cpu().numpy()
            vecs_pred_right = vecs_pred_right.detach().cpu().numpy()
            vecs_pred_face = vecs_pred_face.detach().cpu().numpy()
            vecs_pred = vecs_pred.detach().cpu().numpy()
            vecs_pred_combined = vecs_pred_combined.detach().cpu().numpy()
            # undo scale+translation patch -> pred_space
            verts_pred_left[:, :, [0, 1]] = (verts_pred_left[:, :, [0, 1]] + 1.) * (input_size / 2)
            verts_pred_left[:, :, 2] *= (input_size / 2)
            verts_pred_right[:, :, [0, 1]] = (verts_pred_right[:, :, [0, 1]] + 1.) * (input_size / 2)
            verts_pred_right[:, :, 2] *= (input_size / 2)
            for i_b in range(batch_size):
                image_path = meta[0]['image_path'][i_b]
                # undo transform image -> patch
                try:
                    trans_left = np.concatenate((meta[0]['trans'][i_b], np.array([[0, 0, 1]])))
                    trans_left_inv = np.linalg.inv(trans_left)[:2]
                    trans_right = np.concatenate((meta[1]['trans'][i_b], np.array([[0, 0, 1]])))
                    trans_right_inv = np.linalg.inv(trans_right)[:2]
                except:
                    print(f'Error inverting bbox crop transform')
                    continue
                height_left = meta[0]['height'][i_b].detach().cpu().numpy()
                height_right = meta[1]['height'][i_b].detach().cpu().numpy()
                verts_in_img_left = trans_coords_from_patch_to_org_3d(
                    copy.deepcopy(verts_pred_left[i_b]), height_left, input_size, trans=trans_left_inv)
                verts_in_img_right = trans_coords_from_patch_to_org_3d(
                    copy.deepcopy(verts_pred_right[i_b]), height_right, input_size, trans=trans_right_inv)
                # keep transform
                P_left = estimate_affine_matrix_3d23d(eye_template['left'], verts_in_img_left[:, [1, 0, 2]])
                P_right = estimate_affine_matrix_3d23d(eye_template['right'], verts_in_img_right[:, [1, 0, 2]])
                # keep lms8
                lms8_left = verts_in_img_left[iris_idxs481][:, [1, 0, 2]]
                lms8_right = verts_in_img_right[iris_idxs481][:, [1, 0, 2]]
                # keep predicted data in a dict
                out_dict = {image_path: {
                    'eyes': {
                        'left':  {'P': P_left.astype(np.float16)},
                        'right': {'P': P_right.astype(np.float16)},
                    },
                    'iris_lms8': {
                        'left':  lms8_left.astype(np.float16),
                        'right': lms8_right.astype(np.float16)
                    },
                    'gaze_vec_from_eyes': {
                        'left':  vecs_pred_left[i_b].astype(np.float16),
                        'right': vecs_pred_right[i_b].astype(np.float16),
                        'face':  vecs_pred_face[i_b].astype(np.float16)
                    },
                    'gaze_vec_predicted': vecs_pred[i_b].astype(np.float16),
                    'gaze_vec_combined': vecs_pred_combined[i_b].astype(np.float16)
                }}
                results.update(out_dict)
        elif config.MODE == 'gaze':
            for i_b in range(batch_size):
                image_path = meta[0]['image_path'][i_b]
                # keep predicted data in a dict
                out_dict = {image_path: {
                    'gaze_vec_predicted': vecs_pred[i_b].astype(np.float16),
                }}
                results.update(out_dict)
        else:
            raise NotImplementedError

    # export dict
    path_export = get_final_results_dir(config, config.CFG)
    if not os.path.exists(path_export):
        os.makedirs(str(path_export))
    name_out = f'results_{config.MODE}_mode.pkl'
    path_export_file = os.path.join(path_export, name_out)
    with open(path_export_file, 'wb') as f:
        cPickle.dump(results, f)
    print(f'Exported output in: {path_export_file}')

    return 
