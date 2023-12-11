import argparse

import os
import tqdm
import pickle
import numpy as np
import multiprocessing as mp
from pathlib import Path

import cv2
import torch
import insightface
from insightface.app import FaceAnalysis


# input arguments
parser = argparse.ArgumentParser(description='Face image preprocessing for Gaze Estimation.')
parser.add_argument('--image_base_dir', type=str, required=True, help='The base directory of the images to be preprocessed')
parser.add_argument('--image_paths_list', type=str, help='The path of a .txt file containing the paths of the images to be preprocessed, relatively to the `image_base_dir`')
parser.add_argument('--output_dir', type=str, default='../output/preprocessing', help='The directory to save the preprocessing output')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to run face tracking on, if any GPU is available.')
parser.add_argument('--n_procs', type=int, default=1, help='Number of parallel processes to fit in a single GPU.')
args = parser.parse_args()

gpu_id = args.gpu_id
input_dir = args.image_base_dir
output_dir = args.output_dir
n_procs = args.n_procs

# load insightface model for detection + alignment
app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'landmark_3d_68'])
# app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
app.prepare(ctx_id=gpu_id, det_thresh=0.5, det_size=(224, 224))


def track_faces(path):
    try:
        # load image
        image_cv = cv2.imread(f"{input_dir}/{path}")
        # alignment
        faces = app.get(image_cv)    
        lms68_3D = faces[0]['landmark_3d_68'][:, [1, 0 ,2]]
        lms5 = lms68_3D[[36, 45, 30, 48, 54]][:, [0, 1]]
        lms5[0] = lms68_3D[36:42].mean(axis=0)[[0, 1]]
        lms5[1] = lms68_3D[42:48].mean(axis=0)[[0, 1]]
        # pack results
        item = {path: {'lms68': lms68_3D, 'lms5': lms5}}
    except:
        return {}, path
    return item, path


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    # find image paths
    paths = []
    for root, dirs, files in os.walk(input_dir):
        print(root, len(dirs), len(files))
        files = [os.path.relpath(f"{root}/{f}", input_dir) for f in files]
        paths.extend(files)
    print(f"NUmber of images to be preprocessed: {len(paths)}")
        
    # run tracking
    paths_iter = paths
    num_cpu = n_procs
    pool = mp.Pool(num_cpu)
    results = list(tqdm.tqdm(pool.imap_unordered(track_faces, [path for path in paths_iter]), total=len(paths_iter)))
    pool.close()
    
    # gather all results
    lms_all = {}
    dummy = [lms_all.update(res[0]) for res in results]
    print(f"Number of images successfully preprocessed: {len(lms_all)}")

    # store data in format readable for Gaze Estimation inference
    data = []
    for path, lms in tqdm.tqdm(lms_all.items(), desc='Packing data'):
        lms68_3D = lms['lms68']
        lms5 = lms['lms5']
        item = {
            'name': path,
            'eyes': None,
            'face': {
                # for inference at least one of the bellow sets of face lms must be available
                'xy5': lms5.astype(np.float16),
                'xyz68': lms68_3D.astype(np.float16),
                'head_pose': np.array([0, 0]).astype(np.float16)
            },
            'gaze': None,
        }
        data.append(item)

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/data_face68.pkl", 'wb') as f:
        pickle.dump(data, f)


    

