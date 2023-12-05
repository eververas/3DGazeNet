import cv2
import numpy as np
import menpo.io as mio
from menpo.shape import PointCloud, TriMesh
import matplotlib.pyplot as plt


# --------------------------------------------------------------------

# load mean eyes
eyes_info = mio.import_pickle('../data/eyes3d.pkl')
idxs481 = eyes_info['mask481']['idxs']
tri481 = eyes_info['mask481']['trilist']
iris_idx_481 = eyes_info['mask481']['idxs_iris']
eyel_template = eyes_info['left_points'][idxs481]
eyer_template = eyes_info['right_points'][idxs481]
eyel_template_homo = np.append(eyel_template, np.ones((eyel_template.shape[0],1)), axis=1)
eyer_template_homo = np.append(eyer_template, np.ones((eyer_template.shape[0],1)), axis=1)


# --------------------------------------------------------------------

def view_eyes(fid, image, eyes, crop_to_eyes=False, mode=0):
    '''
    Available modes:
        0: display both eyes and lms8
        1: display only eyes
        2: display only lms8
    '''
    img_crp = image
    eye_l = eyes['left'][:, [0, 1]]
    eye_r = eyes['right'][:, [0, 1]]
    if crop_to_eyes:
        # pointcloud to crop image to
        pc_crop = [eyes['left'], eyes['right']]
        img_crp, trf_crp = image.crop_to_pointcloud_proportion(
            PointCloud(np.concatenate(pc_crop, axis=0)).with_dims([0, 1]),
            0.4, return_transform=True)
        # apply crop transform
        eye_l = trf_crp.pseudoinverse().apply(eyes['left'][:, [0, 1]])
        eye_r = trf_crp.pseudoinverse().apply(eyes['right'][:, [0, 1]])
    lms8_l = eye_l[iris_idx_481] 
    lms8_r = eye_r[iris_idx_481] 
    
    # --- view
    img_crp.view(fid)
    if not mode == 2:
        # left
        TriMesh(eye_l, tri481).view(fid, marker_size=0.01, line_width=0.1, line_colour='r')
        PointCloud(eye_l[iris_idx_481]).view(fid, render_numbering=True)
        # right
        TriMesh(eye_r, tri481).view(fid, marker_size=0.01, line_width=0.1, line_colour='r')
        PointCloud(eye_r[iris_idx_481]).view(fid, render_numbering=True)
    if not mode == 1:
        PointCloud(lms8_l).view(fid, render_numbering=False, marker_face_colour='b')
        PointCloud(lms8_r).view(fid, render_numbering=False, marker_face_colour='b')
        
        
def draw_gaze_from_vector(img_cv, lms68_3D, vector, colour=[255, 0, 0]):
    '''
    Input args:
        path: full image path
        lms68_3D: face lms68
        vector: gaze vector
        colour: colour of the gaze vector
    '''
    # face diag
    lms5 = lms68_3D[[36, 45, 30, 48, 54]][:, [0, 1]]
    lms5[0] = lms68_3D[36:42].mean(axis=0)[[0, 1]]
    lms5[1] = lms68_3D[42:48].mean(axis=0)[[0, 1]]
    diag1 = np.linalg.norm(lms5[0] - lms5[4])
    diag2 = np.linalg.norm(lms5[1] - lms5[3])
    diag = np.max([diag1, diag2])
    # norm weights
    vector_norm = 1.3 * diag
    thickness = int(diag / 10)
    
    # gaze vector eye in image space
    cnt = lms68_3D[30, [1, 0]].astype(np.int32)
    vector_norm = int(np.array(img_cv.shape[1])/4.)
    g_vector = vector[[1, 0]] * np.array([-1., -1.]) * vector_norm
    start_point = cnt[[1, 0]]
    g_point = start_point + g_vector
    # draw gaze vectors
    pt1 = start_point.astype(np.int32)[[1, 0]]
    pt2 = g_point.astype(np.int32)[[1, 0]]
    cv2.arrowedLine(img_cv, pt1, pt2, colour, thickness, tipLength=0.2)
    # display image
    plt.imshow(img_cv[:, :, [2, 1, 0]])