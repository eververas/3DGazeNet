import os
import sys
import cv2
import tqdm
import h5py
import pickle
import numpy as np
import multiprocessing as mp
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

from utils import *


# data set base dir
# path_base = Path('/storage/nfs2/evangelosv/databases/EyeReconstruction/XGaze/xgaze_448/')
path_base = Path('../datasets/XGaze/xgaze_448/')

# paths of all h5py files
mode = 'train'
paths_h5_train = [p for p in (path_base / mode).glob('*.h5')]
print('n h5py files train: {}'.format(len(paths_h5_train)))
mode = 'test'
paths_h5_test = [p for p in (path_base / mode).glob('*.h5')]
print('n h5py files test: {}'.format(len(paths_h5_test)))

# subject ids
mode = 'train'
sub_ids_train = [p.stem.split('.')[0] for p in (path_base / mode).glob('subject*')]
print(f"n train ids: {len(sub_ids_train)}")
mode = 'test'
sub_ids_test = [p.stem.split('.')[0] for p in (path_base / mode).glob('subject*')]
print(f"n test ids: {len(sub_ids_test)}")


# n_frames per subject
# train
n_data_per_sbj_train = {}
for path_h5 in tqdm.tqdm(paths_h5_train):
    fid = h5py.File(str(path_h5), 'r')
    n_data_per_sbj_train[str(path_h5.stem)] = fid['face_patch'].shape[0]
n_data_train = sum([v for v in n_data_per_sbj_train.values()])
print('total frames train: {}'.format(n_data_train))

# test
n_data_per_sbj_test = {}
for path_h5 in tqdm.tqdm(paths_h5_test):
    fid = h5py.File(str(path_h5), 'r')
    n_data_per_sbj_test[str(path_h5.stem)] = fid['face_patch'].shape[0]
n_data_test = sum([v for v in n_data_per_sbj_test.values()])
print('total frames test: {}'.format(n_data_test))

n_data_per_sbj = {}
n_data_per_sbj.update(n_data_per_sbj_train)
n_data_per_sbj.update(n_data_per_sbj_test)


# Export frames from h5py file ---------------------------------------------------------

path_base_export = path_base / 'exported_test'

print('Exporting images from .h5py files...')

def export_images(path):
    try:
        sub_id = path.stem
        mode = path.parent.stem
        path_imgs = path_base_export / mode / 'images'
         # browse images and export
        with h5py.File(str(path), 'r') as fid:
#             for num_i in np.arange(n_data_per_sbj[sub_id]):
            for num_i in np.arange(10):
                image = fid['face_patch'][num_i, :]
                cv2.imwrite(str(path_imgs / sub_id / (str(num_i) + '.jpg')), image)
    except:
        return sub_id, False
    return sub_id, True

# count exported frames 
def count_frames(path):
    sub_id = path.stem
    mode = path.parent.stem
    path_imgs = path_base_export / mode / 'images'
    paths = [p for p in (path_imgs / sub_id).glob('*.jpg')]
    return path.stem, paths

# create export dirs -------------
mode = 'train'
path_imgs = path_base_export / mode / 'images'
paths_sbjs = [path_imgs / key for key in sub_ids_train]
dummy = [os.makedirs(str(path), exist_ok=True) for path in paths_sbjs]
mode = 'test'
path_imgs = path_base_export / mode / 'images'
paths_sbjs = [path_imgs / key for key in sub_ids_test]
dummy = [os.makedirs(str(path), exist_ok=True) for path in paths_sbjs]

# export images -------------------
# train set
paths_iter = paths_h5_train
num_cpu = 50
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(export_images, [path for path in paths_iter]), total=len(paths_iter)))
pool.close()
# paths_true = [r[0] for r in results if r[-1] == True]
# paths_false = [r[0] for r in results if r[-1] == False]
# print(f"Train Set - Successfully exported: {len(paths_true)}/{len(paths_true) + len(paths_false)}")
# test set
paths_iter = paths_h5_test
num_cpu = 50
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(export_images, [path for path in paths_iter]), total=len(paths_iter)))
pool.close()
# paths_true = [r[0] for r in results if r[-1] == True]
# paths_false = [r[0] for r in results if r[-1] == False]
# print(f"Test Set - Successfully exported: {len(paths_true)}/{len(paths_true) + len(paths_false)}")

# count exported images ----------
# train set
paths_iter = paths_h5_train
num_cpu = 50
pool = mp.Pool(num_cpu)
result = list(tqdm.tqdm(pool.imap_unordered(count_frames, paths_iter), total=len(paths_iter)))
pool.close()
mode = 'train'
path_imgs = path_base_export / mode / 'images'
paths_train = sum([r[1] for r in result], [])
paths_train = [str(p.relative_to(path_imgs)) for p in paths_train]
print(f"Train Set - Successfully exported: {len(paths_train)}/{n_data_train}")
# test set
paths_iter = paths_h5_test
num_cpu = 50
pool = mp.Pool(num_cpu)
result = list(tqdm.tqdm(pool.imap_unordered(count_frames, paths_iter), total=len(paths_iter)))
pool.close()
mode = 'test'
path_imgs = path_base_export / mode / 'images'
paths_test = sum([r[1] for r in result], [])
paths_test = [str(p.relative_to(path_imgs)) for p in paths_test]
print(f"Test Set - Successfully exported: {len(paths_test)}/{n_data_test}")


# Lms and Headpose ------------------------------------------------------------------------------
print('Loading face and iris lms...')

# face lms
mode = 'train'
with open(str(path_base_export / mode / 'lms68_3D_all.pkl'), 'rb') as f:
    lms68_3D_all_train = pickle.load(f)
# mode = 'test'
# with open(str(path_base_export / mode / 'lms68_3D_all.pkl'), 'rb') as f:
#     lms68_3D_all_test = pickle.load(f)

# iris lms
mode = 'train'
with open(str(path_base_export / mode / 'lms8_all.pkl'), 'rb') as f:
    lms8_all_train = pickle.load(f)
# mode = 'test'
# with open(str(path_base_export / mode / 'lms8_all.pkl'), 'rb') as f:
#     lms8_all_test = pickle.load(f)


# Annotations dictionary --------------------------------------------------------------------
print('Reading annotations...')
def get_annotations(path_h5):
    try:
        fid = h5py.File(str(path_h5), 'r')
        if 'face_gaze' not in fid.keys():
            fid.close()
            return None
        sub_id = path_h5.stem
        mode = path_h5.parent.stem
        path_imgs = path_base_export / mode / 'images'
        path_imgs_sbj = path_imgs / sub_id
        paths = [p for p in path_imgs_sbj.glob('*.jpg')]
        annotations = {}
        for path in paths:
            num_i = int(path.stem)
            pitchyaws = fid['face_gaze'][num_i, :]
            vector = pitchyaw_to_vector(pitchyaws[None, :])[0]
            cam_index = fid['cam_index'][num_i, 0]
            face_head_pose = fid['face_head_pose'][num_i]
            annotations[str(path.relative_to(path_imgs))] = {
                'gaze': {'pitchyaws': pitchyaws, 'vector': vector},
                'cam_index': cam_index,
                'face_head_pose': face_head_pose}
    except:
        return None
    return annotations

paths_iter = paths_h5_train
num_cpu = 50
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(get_annotations, paths_iter), total=len(paths_iter)))
pool.close()

annotations_train = {}
dummy = [annotations_train.update(res) for res in tqdm.tqdm(results) if res is not None]
print(f"Annotations dict for train set, size: {len(annotations_train)}")


# Fit 3D eyes -----------------------------------------------------------------------------
print('Fitting 3D eyes...')
def fit_eyes(path):
    try:
        pitchyaws = annotations_train[path]['gaze']['pitchyaws']
        vec1 = annotations_train[path]['gaze']['vector'] * np.array([1., -1., 1.])
        vec2 = np.array([0, 0, 1])
        vec = np.cross(vec2, vec1)
        vec = vec / np.linalg.norm(vec)
        angle_norm = np.arccos(np.sum(vec1 * vec2))
        R = Rot.from_rotvec(angle_norm * vec).as_matrix()

        # rotate eyeball
        left_xyz_rot  = (R @ view2rot_axes(eyel_template).T).T
        right_xyz_rot = (R @ view2rot_axes(eyer_template).T).T
        left_xyz_rot  = rot2view_axes(left_xyz_rot)
        right_xyz_rot = rot2view_axes(right_xyz_rot) 

        # image lm8 from model output
        lms8_left = lms8_all_train[path]['left'][:, [0, 1]]
        lms8_right = lms8_all_train[path]['right'][:, [0, 1]]
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

        # get affine transforms -----------------------------------------------------------------------
        Pl = estimate_affine_matrix_3d23d(eyel_template, left_out)
        Pr = estimate_affine_matrix_3d23d(eyer_template, right_out)

        # get head_pose
        head_pose = annotations_train[path]['face_head_pose']

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

paths_iter = paths_train
num_cpu = 60
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(fit_eyes, [p for p in paths_iter]), total=len(paths_iter)))
pool.close()

paths_true = [r[1] for r in results if r[-1]]
paths_false = [r[1] for r in results if r[-1] == False]
print(f"Train Set - Eyes Fitted successfully: {len(paths_true)}/{len(paths_true) + len(paths_false)}")

eyes_all_P = {}
dummy = [eyes_all_P.update(res[0]) for res in tqdm.tqdm(results)]


# Pack data in a format compatible with Gaze Model training -------------------------------------------
print('Packing data...')
def pack_data(path):    
    try:
        name = str(path)
        gaze ={
            'pitchyaws': annotations_train[name]['gaze']['pitchyaws'].astype(np.float16),
            'vector'   : annotations_train[name]['gaze']['vector'].astype(np.float16)}
        eyes = eyes_all_P[name]['eyes']
        lms68_3D = lms68_3D_all_train[name]
        lms5 = lms68_3D[[36, 45, 30, 48, 54]][:, [0, 1]]
        lms5[0] = lms68_3D[36:42].mean(axis=0)[[0, 1]]
        lms5[1] = lms68_3D[42:48].mean(axis=0)[[0, 1]]
        head_pose = annotations_train[name]['face_head_pose']
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

paths_iter = paths_train
num_cpu = 50
pool = mp.Pool(num_cpu)
results = list(tqdm.tqdm(pool.imap_unordered(pack_data, [path for path in paths_iter]), total=len(paths_iter)))
pool.close()

paths_true = [r[0] for r in results if r[-1]]
paths_false = [r[0] for r in results if r[-1] == False]
print(f"Train Set - Packed successfully: {len(paths_true)}/{len(paths_true) + len(paths_false)}")

results_train = [r[0] for r in tqdm.tqdm(results) if r[-1]]
print(f"Number of samples packed: {len(results_train)}")

# export processed data
mode = 'train'
path_export_data = path_base_export / mode / 'data_for_model'
os.makedirs(str(path_export_data), exist_ok=True)
with open(str(path_export_data / 'train_gaze_eyes3D_face68.pkl'), 'wb') as f:
    pickle.dump(results_train, f)