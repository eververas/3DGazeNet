import os
import cv2
import copy
import torch
import numpy as np
import _pickle as cPickle
from matplotlib import pyplot as plt


PATH_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_eyes3d(fname_eyes3d=PATH_BASE + '/data/eyes3d.pkl'):
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

def kpt5_from_68(kpt):
    kpt5 = kpt[[36, 45, 30, 48, 54]][:, [1, 0]]
    kpt5[0] = kpt[36:42].mean(axis=0)[[1, 0]]
    kpt5[1] = kpt[42:48].mean(axis=0)[[1, 0]]
    return kpt5


def get_face_diag_from_lms5(lms5):
    diag1 = np.linalg.norm(lms5[0] - lms5[4])
    diag2 = np.linalg.norm(lms5[1] - lms5[3])
    return np.max([diag1, diag2])


def get_bbox_info(lms5, element_str):
    diag = get_face_diag_from_lms5(lms5)
    # width
    crop_len = int(diag / 5)
    if element_str == 'face':
        crop_len = 2. * diag
    width = crop_len
    height = crop_len
    # center
    cnt = lms5[2] 
    if element_str == 'left_eye':
        cnt = lms5[1]
    if element_str == 'right_eye':
        cnt = lms5[0]
    center_x = cnt[0]
    center_y = cnt[1]
    return center_x, center_y, width, height


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


def draw_gaze(image, lms, vector, colour=[0, 0, 255]):
    '''
    Input args:
        image: cv2 image
        lms: face lms68 or lms5 in cv space
        vector: gaze vector
        colour: colour of the gaze vector
    '''
    # face diag
    lms5 = lms.copy()
    if lms5.shape[0] == 68:
        lms5 = kpt5_from_68(lms5)
    diag = get_face_diag_from_lms5(lms5)
    # norm weights
    vector_norm = 1.3 * diag
    thickness = int(diag / 10)
    
    # gaze vector eye in image space
    cnt = lms5[2].astype(np.int32)
    g_vector = -vector * vector_norm
    start_point = cnt
    g_point = start_point + g_vector[:2]
    # draw gaze vectors
    pt1 = start_point.astype(np.int32)
    pt2 = g_point.astype(np.int32)
    image = cv2.arrowedLine(image, pt1, pt2, colour, thickness, tipLength=0.2)
    return image


idxs_ring0 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
idxs_ring1 = [240, 112, 80, 48, 16, 304, 336, 368, 442, 480, 426, 352, 320, 288, 0, 32, 64, 96, 224]
idxs_ring2 = [248, 120, 88, 56, 24, 312, 344, 376, 433, 480, 421, 360, 328, 296, 8, 40, 72, 104, 232]
idxs_ring_iris = [224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254]

def draw_eyes(image, lms, eyes, draw_eyeball=True, draw_iris=True, draw_rings=True, colour=[178, 255, 102]):
    '''
    Input args:
        image: cv2 image
        lms: face lms68 or lms5
        eyes: dictionary including left and right eyes as np arrays
        colour: colour to use for drawing the eyes
    '''
    # colour_iris = colour
    colour_iris = [0, 255, 255]
    colour_eyeball = colour
    colour_rings = colour

    # face diag
    lms5 = lms.copy()
    if lms5.shape[0] == 68:
        lms5 = kpt5_from_68(lms5)
    diag = get_face_diag_from_lms5(lms5)
    thickness_iris = 2 if max(int(diag / 100), 1) < 3 else 3
    thickness_eyeball = 2 if max(int(diag / 100), 1) < 3 else 3
    thickness_rings = 1 if max(int(diag / 100), 1) < 3 else 2
    
    # draw eyeballs
    if draw_eyeball:
        for side, eye in eyes.items():
            # eye = eye[:, [1, 0]]
            ring1 = eye[idxs_ring1][:, :2]
            ring2 = eye[idxs_ring2][:, :2]
            range1 = ring1[:, 1].max() - ring1[:, 1].min()
            range2 = ring2[:, 0].max() - ring2[:, 0].min()
            radius = int(range1/4 + range2/4)
            cnt = eye[:32].mean(axis=0)
            image = cv2.circle(
                image, tuple(cnt[:2].astype(np.int32).tolist()), radius, colour_eyeball, thickness_eyeball)
    # draw rings
    if draw_rings:
        for side, eye in eyes.items():
            # eye = eye[:, [1, 0]]
            iris8 = eye[idxs_ring1][:, :2]
            for i_idx in range(iris8.shape[0]-1):
                pt1 = tuple(iris8[i_idx].astype(np.int32).tolist())
                pt2 = tuple(iris8[i_idx + 1].astype(np.int32).tolist())
                image = cv2.line(image, pt1, pt2, colour_rings, thickness_rings)
        for side, eye in eyes.items():
            # eye = eye[:, [1, 0]]
            iris8 = eye[idxs_ring2][:, :2]
            for i_idx in range(iris8.shape[0]-1):
                pt1 = tuple(iris8[i_idx].astype(np.int32).tolist())
                pt2 = tuple(iris8[i_idx + 1].astype(np.int32).tolist())
                image = cv2.line(image, pt1, pt2, colour_rings, thickness_rings)
    # draw irises
    if draw_iris:
        for side, eye in eyes.items():
            # eye = eye[:, [1, 0]]
            iris8 = eye[idxs_ring_iris][:, :2]
            for i_idx in range(iris8.shape[0]):
                pt1 = tuple(iris8[i_idx].astype(np.int32).tolist())
                pt2 = tuple(iris8[(i_idx + 1) % 16].astype(np.int32).tolist())
                image = cv2.line(image, pt1, pt2, colour_iris, thickness_iris) 
    return image


def draw_results(img_cv, lms5, gaze_dict):
    # render gaze for face in image
    if gaze_dict['verts_eyes'] is None:
        img_cv = draw_gaze(img_cv, lms5, gaze_dict['gaze_combined'])
    else:
        img_cv = draw_eyes(img_cv, lms5, gaze_dict['verts_eyes'], draw_rings=False)
        img_cv = draw_gaze(img_cv, lms5, gaze_dict['gaze_combined'])

    return img_cv



