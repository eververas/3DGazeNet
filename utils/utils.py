import os

import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt


def get_gaze_pitchyaws_from_vectors(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan(vectors[:, 0] / vectors[:, 2])  # phi
    if n == 1:
        out = out[0]
    return out


def get_gaze_pitchyaws_from_eyes(eye, iris_lms_idx):
    p_iris = eye[iris_lms_idx] - eye[:32].mean(axis=0)
    vector = p_iris.mean(axis=0)[np.newaxis, :]

    out = np.empty(2)
    vector = np.divide(vector, np.linalg.norm(vector, axis=1).reshape(1, 1))
    out[0] = np.arcsin(vector[:, 1])  # theta
    out[1] = np.arctan(vector[:, 0] / vector[:, 2])  # phi
    return out, vector


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

    shape = img.shape[:2]  # current shape [height, width]
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


def draw_gaze(img, diag, gaze_dict, color=(255, 0, 0), from_angle='gaze_angle_from_vector'):
    ry, rx = gaze_dict[from_angle]
    dx = 0.4 * diag * np.sin(-rx)
    dy = 0.4 * diag * np.sin(-ry)
    pt1 = gaze_dict['eye_center']
    pt2 = np.array((pt1[0] + dx, pt1[1] + dy)).astype(np.int32)
    cv2.arrowedLine(img, pt1, pt2, color, 2)

    return img


def draw_results(img, bbox, gaze):
    # render gaze + borders for each face in image
    # get gaze arrow
    diag = np.linalg.norm(np.array([bbox[1], bbox[2]]) - np.array([bbox[3], bbox[0]]))
    for eye_str in ['left_eye', 'right_eye']:
        gaze_dict = gaze[eye_str]
        img = draw_gaze(img, diag, gaze_dict, color=(0, 0, 255), from_angle='gaze_angle_from_vector')
        # img = draw_gaze(img, diag, gaze_dict, color=(1, 190, 200), from_angle='gaze_angle_from_eyes')
    return img
