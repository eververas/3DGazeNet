import os
import cv2
import tqdm
import pickle
import numpy as np
import multiprocessing as mp
from pathlib import Path
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as Rot

from utils import *

def get_path(i):
    rec = meta['recordings'][0][meta['recording'][0][i]][0]
    sbj = str(meta['person_identity'][0][i]).zfill(6)
    frame_id = str(meta['frame'][0][i]).zfill(6)
    path = f"{rec}/head/{sbj}/{frame_id}.jpg"
    return path

# Read dataset meta
# path_base = Path('/storage/nfs2/evangelosv/databases/EyeReconstruction/gaze360/')
path_base = Path('../datasets/XGaze/xgaze_448/')
path_imgs = path_base / 'images'
meta = loadmat(str(path_base / 'metadata.mat'))

n_paths = meta['recording'][0].shape[0]
paths_db = [get_path(i) for i in np.arange(n_paths)]
print(f"Total number of samples in dataset: {len(paths_db)}")

# Read paths with filtered headpose
paths_train = np.loadtxt(str(path_base / 'paths_train_hp90.txt'), np.str_).tolist()
paths_test = np.loadtxt(str(path_base / 'paths_test_hp90.txt'), np.str_).tolist()
paths = paths_train + paths_test
print(f"Total number of samples with headpose <= 90 degrees: train: {len(paths_train)}, test: {len(paths_test)}, all: {len(paths)}")

# Lms and Headpose ------------------------------------------------------------------------------

# tracked with insightface
with open(str(path_base / 'lms68_3D_wbb.pkl'), 'rb') as f:
    lms68_wbb = pickle.load(f)
with open(str(path_base / 'lms68_3D_wobb.pkl'), 'rb') as f:
    lms68_wobb = pickle.load(f)
lms68_all = lms68_wbb.copy()
lms68_all.update(lms68_wobb)
print(f"face lms of images with given bboxes: {len(lms68_wbb)}, without given bboxes: {len(lms68_wobb)}, all: {len(lms68_all)}")

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
def get_annotations(idx):
    path = get_path(idx)
    vector = meta['gaze_dir'][idx] * (1, 1, -1)
    pitchyaws = vector_to_pitchyaw(vector[None, :])[0]
    try:
        face_head_pose = headpose_all[path]
    except:
        face_head_pose = np.zeros(3)
    annotation = {}
    annotation[path] = {
        'gaze': {'pitchyaws': pitchyaws, 'vector': vector},
        'face_head_pose': face_head_pose}
    return annotation
    
paths_iter = np.arange(len(paths_db))
num_cpu = 50
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(get_annotations, paths_iter), total=len(paths_iter)))
pool.close()

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
    
        # scale to edge length ------------------------------------------------------------------------
        # lms5
        lms68_3D = lms68_all[path][:, [0, 1, 2]]
        lms5 = lms68_3D[[36, 45, 30, 48, 54]][:, [0, 1]]
        lms5[0] = lms68_3D[36:42].mean(axis=0)[[0, 1]]
        lms5[1] = lms68_3D[42:48].mean(axis=0)[[0, 1]]
        diag1 = np.linalg.norm(lms5[0] - lms5[4])
        diag2 = np.linalg.norm(lms5[1] - lms5[3])
        diag = np.max([diag1, diag2])
        # left
        diffs = left_out[tri481][:, :] - left_out[tri481][:, [1, 2, 0]]
        diff_l = np.linalg.norm(diffs, axis=2).mean() / diag
        scl_l = 1.
        if diff_l >= 0.03:
            scl_l = 0.03 / diff_l
        if diff_l <= 0.02:
            scl_l = 0.02 / diff_l
        cnt_l = left_out[iris_idx_481].mean(axis=0)
        left_out -= left_out[:32].mean(axis=0)
        left_out *= scl_l
        left_out += - left_out[iris_idx_481].mean(axis=0) + cnt_l
        # right
        diffs = right_out[tri481][:, :] - right_out[tri481][:, [1, 2, 0]]
        diff_r = np.linalg.norm(diffs, axis=2).mean() / diag
        scl_r = 1.
        if diff_r >= 0.03:
            scl_r = 0.03 / diff_r
        if diff_r <= 0.02:
            scl_r = 0.02 / diff_r
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

# pack train set -------------------------
paths_iter = paths_train
num_cpu = 50
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(pack_data, [path for path in paths_iter]), total=len(paths_iter)))
pool.close()

paths_true = [r[0] for r in results if r[-1]]
paths_false = [r[0] for r in results if r[-1] == False]
print(f"Train Set - Packed successfully: {len(paths_true)}/{len(paths_true) + len(paths_false)}")

results_train = [r[0] for r in results if r[-1]]
print(f"Number of samples packed: {len(results_train)}")

# export processed data
path_export_data = path_base / 'data_for_model_2'
os.makedirs(str(path_export_data), exist_ok=True)
with open(str(path_export_data / 'train_gaze_eyes3D_face68.pkl'), 'wb') as f:
    pickle.dump(results_train, f)
print(f"Data exported in: {path_export_data / 'train_gaze_eyes3D_face68.pkl'}")

# pack test set -------------------------
paths_iter = paths_test
num_cpu = 50
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(pack_data, [path for path in paths_iter]), total=len(paths_iter)))
pool.close()

paths_true = [r[0] for r in results if r[-1]]
paths_false = [r[0] for r in results if r[-1] == False]
print(f"Test Set - Packed successfully: {len(paths_true)}/{len(paths_true) + len(paths_false)}")

results_test = [r[0] for r in results if r[-1]]
print(f"Number of samples packed: {len(results_test)}")

# export processed data
path_export_data = path_base / 'data_for_model_2'
os.makedirs(str(path_export_data), exist_ok=True)
with open(str(path_export_data / 'test_gaze_eyes3D_face68.pkl'), 'wb') as f:
    pickle.dump(results_test, f)
print(f"Data exported in: {path_export_data / 'test_gaze_eyes3D_face68.pkl'}")


