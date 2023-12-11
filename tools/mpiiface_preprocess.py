import os
import cv2
import tqdm
import h5py
import pickle
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

from utils import *

# path_base = Path('/storage/nfs2/evangelosv/databases/EyeReconstruction/MPIIFaceGaze/')
path_base = Path('datasets/MPIIFaceGaze')
path_db = path_base / 'dataset'
path_imgs = path_base / 'images_2'
paths_mat = [p for p in path_db.rglob('*.mat')]

# Export images from .mat files ----------------------------------------------------------------
print('Exporting images from .mat files...')

# create folders
for path in paths_mat:
    if not (path_imgs / path.stem).exists():
        os.makedirs(str(path_imgs / path.stem))

# export images
def export_images(path_mat):
    try:
        with h5py.File(str(path_mat), 'r') as mat:
            n_samples = mat['Data']['data'].shape[0]
            for i in range(n_samples):
                try:
                    image = np.transpose(mat['Data']['data'][i], axes=[1, 2, 0])
                    cv2.imwrite(str(path_imgs / path_mat.stem / (str(i) + '.jpg')), image)
                except:
                    print('error exporting image')
                    pass
    except:
        return path_mat, False
    return path_mat, True

paths_iter = paths_mat
num_cpu = 50
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(export_images, paths_iter), total=len(paths_iter)))
pool.close()

# count images
paths = []
for path_mat in paths_mat:
    paths += [str(p.relative_to(path_imgs)) for p in (path_imgs / path_mat.stem).glob('*.jpg')]
print(f"Successfully exported: {len(paths)}/{len(paths_mat)*3000}")


# Lms and Headpose ------------------------------------------------------------------------------
print('Loading face and iris lms...')
# face lms
with open(str(path_base / 'lms68_3D_all.pkl'), 'rb') as f:
    lms68_all = pickle.load(f)
# iris lms
with open(str(path_base / 'lms8_all.pkl'), 'rb') as f:
    lms8_all = pickle.load(f)
# head pose from lms68
print('Calculating headpose from 3D face landmarks...')
def get_headpose(path):
    try:
        lms68_3D = lms68_all[path]
        angles = headpose_from_lms68(lms68_3D)
        item = {path: angles}
    except:
        return {}, path, False
    return item, path, True

paths_iter = paths
num_cpu = 50
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(get_headpose, [p for p in paths_iter]), total=len(paths_iter)))
pool.close()

headpose_all = {}
dummy = [headpose_all.update(res[0]) for res in results]


# Annotations dictionary ------------------------------------------------------------------------
print('Reading annotations...')
def load_annotations(path_mat):
    try:
        with h5py.File(str(path_mat), 'r') as mat:
            n_samples = mat['Data']['data'].shape[0]
            annotations = {}
            for i in range(n_samples):
                gaze_pitchyaw = mat['Data']['label'][i][0:2]
                gaze_vector = pitchyaw_to_vector(gaze_pitchyaw[None, :])[0]
                headpose = mat['Data']['label'][i][2:4].astype(np.float16)
                lms6 = mat['Data']['label'][i][4:].reshape((6, 2)).astype(np.float16)
                annotations[str(Path(path_mat.stem) / (str(i) + '.jpg'))] = {
                    'gaze': {
                        'pitchyaws': gaze_pitchyaw, 'vector': gaze_vector},
                    'face_head_pose': headpose,
                    'lms6': lms6
                }
    except:
        return {}, path_mat, False
            
    return annotations, path_mat, True

paths_iter = paths_mat
num_cpu = 50
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(load_annotations, paths_iter), total=len(paths_iter)))
pool.close()
results = [res[0] for res in results]

annotations = {}
dummy = [annotations.update(res) for res in results]
print(f"Annotations read for {len(annotations)} samples")


# Fit 3D eyes ------------------------------------------------------------------------------------
print('Fitting 3D eyes...')
def fit_eyes(path):
    try:
        pitchyaws = annotations[path]['gaze']['pitchyaws']
        vec1 = annotations[path]['gaze']['vector'] * np.array([1., -1., 1.])
        vec2 = np.array([0, 0, 1])
        vec = np.cross(vec2, vec1)
        vec = vec / np.linalg.norm(vec)
        angle_norm = np.arccos(np.sum(vec1 * vec2))
        R = Rot.from_rotvec(angle_norm * vec).as_matrix()
    
        # rotate
        left_xyz_rot  = (R @ view2rot_axes(eyel_template).T).T
        right_xyz_rot = (R @ view2rot_axes(eyer_template).T).T
        left_xyz_rot  = rot2view_axes(left_xyz_rot)
        right_xyz_rot = rot2view_axes(right_xyz_rot) 
    
        # image lm8 from model output
        lms8_left = lms8_all[path]['left'][:, [0, 1]]
        lms8_right = lms8_all[path]['right'][:, [0, 1]]
        lms8_left_cnt = lms8_left.mean(axis=0)
        lms8_right_cnt = lms8_right.mean(axis=0)
    
        # fit left ---------------------------------------------------------------------------------
        left_xyz_rot[:, [0, 1]] -= left_xyz_rot[iris_idx_481][:, [0, 1]].mean(axis=0)
        # scale 
        source = left_xyz_rot[:, [0, 1]][iris_idx_481]
        target = lms8_left - lms8_left_cnt
        scale = np.linalg.norm(target) / np.linalg.norm(source)
        left_xyz_rot_alg = scale * left_xyz_rot
        # translate
        left_xyz_rot_alg[:, [0, 1]] += lms8_left_cnt
        left_out = left_xyz_rot_alg.astype(np.float32)
    
        # fit right --------------------------------------------------------------------------------
        right_xyz_rot[:, [0, 1]] -= right_xyz_rot[iris_idx_481][:, [0, 1]].mean(axis=0)
        # scale
        source = right_xyz_rot[:, [0, 1]][iris_idx_481]
        target = lms8_right - lms8_right_cnt
        scale = np.linalg.norm(target) / np.linalg.norm(source)
        right_xyz_rot_alg = scale * right_xyz_rot
        # translate
        right_xyz_rot_alg[:, [0, 1]] += lms8_right_cnt
        right_out = right_xyz_rot_alg.astype(np.float32)
        
        
        # scale to edge length ----------------------------------------------------------------------
        diffs = left_out[tri481][:, :] - left_out[tri481][:, [1, 2, 0]]
        diff_l = np.linalg.norm(diffs, axis=2).mean()
        diffs = right_out[tri481][:, :] - right_out[tri481][:, [1, 2, 0]]
        diff_r = np.linalg.norm(diffs, axis=2).mean()
        if diff_l >= 10. or diff_r >= 10: # filter out black images?
            return {}, path, False
        if diff_l < 4 or diff_r < 4: # filter out black images?
            return {}, path, False
        if diff_l >= 8.5:
            scl_l = 8.5 / diff_l 
            cnt_l = left_out[iris_idx_481].mean(axis=0)
            left_out -= left_out[:32].mean(axis=0)
            left_out *= scl_l
            left_out += - left_out[iris_idx_481].mean(axis=0) + cnt_l
        if diff_r >= 8.5:
            scl_r = 8.5 / diff_r
            cnt_r = right_out[iris_idx_481].mean(axis=0)
            right_out -= right_out[:32].mean(axis=0)
            right_out *= scl_r
            right_out += - right_out[iris_idx_481].mean(axis=0) + cnt_r
        
        # get affine transforms -----------------------------------------------------------------------
        Pl = estimate_affine_matrix_3d23d(eyel_template, left_out)
        Pr = estimate_affine_matrix_3d23d(eyer_template, right_out)
    
        # get head_pose
        head_pose = annotations[path]['face_head_pose']
    
        item = {str(path): {
            'eyes': {
                'left' : {'P': Pl.astype(np.float16)},
                'right': {'P': Pr.astype(np.float16)},
            },
            'face': {
                'gaze': pitchyaws.astype(np.float16),
                'head_pose': head_pose.astype(np.float16)
            }
        }}
    except:
        return {}, path, False
    return item, path, True

paths_iter = paths
num_cpu = 60
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(fit_eyes, [p for p in paths_iter]), total=len(paths_iter)))
pool.close()

paths_true = [r[1] for r in results if r[-1]]
paths_false = [r[1] for r in results if r[-1] == False]
print(f"Eyes Fitted successfully: {len(paths_true)}/{len(paths_true) + len(paths_false)}")

eyes_all_P = {}
dummy = [eyes_all_P.update(res[0]) for res in results]


# Pack data in a format compatible with Gaze Model training -------------------------------------------
print('Packing data...')
def pack_data(path):
    try:
        name = str(path)
        gaze ={
            'pitchyaws': annotations[name]['gaze']['pitchyaws'].astype(np.float16),
            'vector'   : annotations[name]['gaze']['vector'].astype(np.float16)}
        eyes = eyes_all_P[name]['eyes']
        lms68_3D = lms68_all[name][:, [0, 1, 2]]
        lms5 = lms68_3D[[36, 45, 30, 48, 54]][:, [0, 1]]
        lms5[0] = lms68_3D[36:42].mean(axis=0)[[0, 1]]
        lms5[1] = lms68_3D[42:48].mean(axis=0)[[0, 1]]
        head_pose = annotations[name]['face_head_pose']
        face = {
            'xy5': lms5.astype(np.float16),
            'xyz68': lms68_3D.astype(np.float16),
            'head_pose': head_pose.astype(np.float16)}
        item = {
            'name': name,
            'eyes': eyes,
            'face': face,
            'gaze':{
                'face': gaze,
                'left': gaze,
                'right': gaze,
            }
        }
    except:
        return {}, path, False
    return item, path, True

paths_iter = paths
num_cpu = 50
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(pack_data, [path for path in paths_iter]), total=len(paths_iter)))
pool.close()

paths_true = [r[0] for r in results if r[-1]]
print(f"Train Set - Packed successfully: {len(paths_true)}/{len(paths)}")
results = [r[0] for r in results if r[-1]]
print(f"Number of samples packed: {len(results)}")

# export processed data
path_export_data = path_base / 'data_for_model_2'
os.makedirs(str(path_export_data), exist_ok=True)
with open(str(path_export_data / 'all_gaze_eyes3D_face68.pkl'), 'wb') as f:
    pickle.dump(results, f)
print(f"Data exported in: {path_export_data / 'all_gaze_eyes3D_face68.pkl'}")