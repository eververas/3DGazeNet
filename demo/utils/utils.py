import os
import cv2
import copy
import torch
import numpy as np
import _pickle as cPickle
from matplotlib import pyplot as plt


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
        'left': eyes3d['left_points'][idxs481],
        'right': eyes3d['right_points'][idxs481]
    }
    eye_template_homo = {
        'left': np.append(eye_template['left'], np.ones((eyel_template.shape[0], 1)), axis=1),
        'right': np.append(eye_template['right'], np.ones((eyer_template.shape[0], 1)), axis=1)
    }
    eyes3d_dict = {
        'iris_idxs': iris_idxs, 
        'idxs481': idxs481, 
        'iris_idxs481': iris_idxs481, 
        'idxs288': idxs288,
        'iris_idxs288': iris_idxs288,
        'trilist_eye': trilist_eye, 
        'eye_template': eye_template,
        'eye_template_homo': eye_template_homo
    }
    return eyes3d_dict


def points_to_vector(points, iris_lms_idx):
    back = points[:, np.arange(32)].mean(axis=1, keepdim=True) # (B, 1, 3)
    front = points[:, iris_lms_idx].mean(axis=1, keepdim=True) # (B, 1, 3)
    vec = front - back
    vec = vec / torch.norm(vec, dim=2, keepdim=True)  # (B, 1, 3)
    return torch.squeeze(vec, dim=1)


def show_result(img, bboxes=None, keypoints=None, gaze=None, title=None):
    import copy
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    """Draw `result` over `img` and plot it on local Pycharm.
        This function is used for internal debugging purposes only.
    Args:
        img (str or Numpy): The image to be displayed.
        bboxes (Numpy or tuple): The bboxes to draw over `img`.
        keypoints (Numpy): The available keypoints to draw over `img`.
    Returns:
        None
    """
    if isinstance(img, str):
        # img = np.asarray(Image.open(img))
        img = cv2.imread(img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        img = copy.deepcopy(img)
    if title is None:
        title = 'test input img with bboxes and keypoints (if available)'
    # draw bounding boxes
    if bboxes is not None:
        for j, _bboxes in enumerate(bboxes):
            left_top = (bboxes[j, 0], bboxes[j, 1])
            right_bottom = (bboxes[j, 2], bboxes[j, 3])
            cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), thickness=1)
    # draw keypoints
    if keypoints is not None:
        for annot in keypoints:
            cor_x, cor_y = int(annot[0]), int(annot[1])
            cv2.circle(img, (cor_x, cor_y), 1, (255, 0, 0), 1)
    if gaze is not None:
        ry, rx = gaze[0]
        eye_center = gaze[1]
        dx = 25 * np.sin(-rx)
        dy = 25 * np.sin(-ry)
        pt2 = np.array((eye_center[0] + dx, eye_center[1] + dy)).astype(np.int32)
        cv2.arrowedLine(img, eye_center.astype(np.int32), pt2, (255, 0, 0), 2)
    # plot the result
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255)  # We expect image to be bgr and to 0-255
    plt.title(title)
    plt.tight_layout()
    plt.show(block=True)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov5

    shape = img_cv.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def trans_verts_from_patch_to_org(verts_in_patch, org_width, patch_width, trans=None):
    verts_in_org = verts_in_patch.copy()
    verts_in_org[:, :2] = affine_transform_array(verts_in_patch[:, :2], trans)
    verts_in_org[:, 2] = verts_in_patch[:, 2] / patch_width * org_width
    return verts_in_org

def affine_transform_array(verts, P):
    verts = np.concatenate((verts, np.ones((verts.shape[0], 1))), axis=1)
    new_verts = (P @ verts.T).T # (N, 2)
    return new_verts[:, :2]


def draw_gaze(img_cv, diag, gaze_dict):
    thickness = int(8 * diag / img_cv.shape[0])
    vector_norm = int(70 * diag / img_cv.shape[0])

    cnt = gaze_dict['centers']['face']
    gaze = gaze_dict['gaze_combined'] if gaze_dict['gaze_combined'] is not None else gaze_dict['gaze']
    # gaze vector in image space
    g_vector = - gaze * vector_norm
    g_point = cnt + g_vector[:2]
    # draw gaze vector
    pt1 = cnt.astype(np.int32)
    pt2 = g_point.astype(np.int32)
    cv2.arrowedLine(img_cv, pt1, pt2, [0, 0, 255], thickness, tipLength=0.2)

    return img_cv

def draw_eyes(img_cv, diag, gaze_dict, face_elem='face'):
    thickness_eyes = int(3 * diag / img_cv.shape[0])
    thickness_gaze = int(5 * diag / img_cv.shape[0])
    vector_norm = int(70 * diag / img_cv.shape[0])
    iris_idxs = gaze_dict['iris_idxs']

    # draw eyeballs
    for side, eye in gaze_dict['verts_eyes'].items():
        cnt = eye[:32].mean(axis=0)
        radius = int(np.linalg.norm(cnt - eye[0]))
        img_cv = cv2.circle(
            img_cv, tuple(cnt[:2].astype(np.int32).tolist()), radius, [155, 0, 0], thickness_eyes)
    # draw irises
    for side, eye in gaze_dict['verts_eyes'].items():
        iris8 = eye[iris_idxs][:, :2]
        for i_idx in range(iris8.shape[0]):
            pt1 = tuple(iris8[i_idx].astype(np.int32).tolist())
            pt2 = tuple(iris8[(i_idx + 1) % 8].astype(np.int32).tolist())
            img_cv = cv2.line(img_cv, pt1, pt2, [155, 0, 0], thickness_eyes)
    # draw gaze vectors
    for side, eye in gaze_dict['verts_eyes'].items():
        # gaze vector eye in image space
        # g_vector = - gaze_dict['gaze_from_eyes'][side] * vector_norm
        g_vector = - gaze_dict['gaze_combined'] * vector_norm
        g_point = eye[iris_idxs].mean(axis=0) + g_vector
        # draw gaze vectors
        pt1 = eye[iris_idxs].mean(axis=0)[:2].astype(np.int32)
        pt2 = g_point[:2].astype(np.int32)
        cv2.arrowedLine(img_cv, pt1, pt2, [0, 0, 255], thickness_gaze, tipLength=0.2)

    return img_cv


def draw_results(img_cv, bbox, gaze_dict):
    # render gaze for face in image
    diag = np.linalg.norm(np.array([bbox[1], bbox[2]]) - np.array([bbox[3], bbox[0]]))
    center = gaze_dict['centers']['face']
    
    if gaze_dict['verts_eyes'] is None:
        img_cv = draw_gaze(img_cv, diag, gaze_dict)
    else:
        img_cv = draw_eyes(img_cv, diag, gaze_dict)
        img_cv = draw_gaze(img_cv, diag, gaze_dict)

    return img_cv



