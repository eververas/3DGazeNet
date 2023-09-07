from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def estimate_affine_matrix_3d23d(X, Y):
    ''' Using least-squares solution
    Args:
        X: [n, 3]. 3d points(fixed)
        Y: [n, 3]. corresponding 3d points(moving). Y = PX
    Returns:
        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).
    '''
    X_homo = np.hstack((X, np.ones([X.shape[0], 1])))  # n x 4
    P = np.linalg.lstsq(X_homo, Y, rcond=-1)[0].T  # Affine matrix. 3 x 4
    return P

def compose_affine_transforms(trans2, trans1):
    """Computes a composite transform s.t. trans(x) = trans2(trans1(x))"""
    return np.matmul(trans2, np.vstack([trans1, np.float64([0, 0, 1])]))

def trans_coords_from_patch_to_org(
        coords_in_patch, patch_width=None, patch_height=None,
        c_x=None, c_y=None, bb_width=None, bb_height=None, trans=None):
    coords_in_org = coords_in_patch.copy()
    if trans is None:
        trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, 0, inv=True)
    coords_in_org[:, :2 ]= affine_transform_array(coords_in_patch[:, :2], trans)
    return coords_in_org

def trans_coords_from_patch_to_org_3d(
        coords_in_patch, rect_3d_width, patch_width,
        patch_height=None, c_x=None, c_y=None, bb_width=None, bb_height=None, trans=None):
    res_img = trans_coords_from_patch_to_org(
        coords_in_patch, patch_width, patch_height, c_x, c_y, bb_width, bb_height, trans)
    res_img[:, 2] = coords_in_patch[:, 2] / patch_width * rect_3d_width
    return res_img

def get_input_and_transform(center, width_height, cv_img_numpy, crop_size, rotation, do_flip):
    img_height, img_width, img_channels = cv_img_numpy.shape
    if do_flip:
        cv_img_numpy = cv_img_numpy[:, ::-1, :]
        center[0] = img_width - center[0] - 1
    trans = gen_trans_from_patch_cv_tuples(center, width_height, crop_size, rotation, inv=False)
    input = cv2.warpAffine(cv_img_numpy, trans, tuple(crop_size), flags=cv2.INTER_LINEAR)

    return trans, input

def gen_trans_from_patch_cv(c_x, c_y, src_w, src_h, dst_w, dst_h, rot, inv=False):
    def rotate_2d(pt_2d, rot_rad):
        x = pt_2d[0]
        y = pt_2d[1]
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        xx = x * cs - y * sn
        yy = x * sn + y * cs
        return np.array([xx, yy], dtype=np.float32)

    rot_rad = np.pi * rot / 180
    #
    src_center = np.array([c_x, c_y], dtype=np.float32)
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
    #
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def gen_trans_from_patch_cv_tuples(center_xy, src_wh, dst_wh, rot, inv=False):
    return gen_trans_from_patch_cv(center_xy[0], center_xy[1],
                                   src_wh[0], src_wh[1],
                                   dst_wh[0], dst_wh[1],
                                   rot, inv)

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def affine_transform_array(verts, P):
    verts = np.concatenate((verts, np.ones((verts.shape[0], 1))), axis=1)
    new_verts = (P @ verts.T).T # (N, 2)
    return new_verts[:, :2]
