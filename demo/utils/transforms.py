import numpy as np
import cv2


def _rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_w, src_h, dst_w, dst_h, rot,
                            inv=False):
    rot_rad = np.pi * rot / 180
    #
    src_center = np.array([c_x, c_y], dtype=np.float32)
    src_downdir = _rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = _rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
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


def get_input_and_transform(center, width_height, cv_img_numpy, crop_size, rotation, do_flip):
    img_height, img_width, img_channels = cv_img_numpy.shape
    if do_flip:
        cv_img_numpy = cv_img_numpy[:, ::-1, :]
        center[0] = img_width - center[0] - 1
    trans = gen_trans_from_patch_cv_tuples(center, width_height, crop_size, rotation, inv=False)
    input = cv2.warpAffine(cv_img_numpy, trans, tuple(crop_size), flags=cv2.INTER_LINEAR)
    return trans, input


# methods to calculate gaze angles and 3D-vector
def angles_from_vec(vec):
    x, y, z = -vec[2], vec[1], -vec[0]
    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x ** 2 + y ** 2), z) - np.pi / 2
    theta_x, theta_y = phi * 180. / np.pi, theta * 180 / np.pi
    return theta_x, theta_y


def get_gaze_pitchyaws_from_vectors(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan(vectors[:, 0] / vectors[:, 2])  # phi
    return out
