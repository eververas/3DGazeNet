import tqdm
import torch
import torch.backends.cudnn as cudnn
import os
import argparse

from utils import config as cfg, update_config, get_logger, Timer, VideoLoader, VideoSaver, show_result, draw_results
from models import FaceDetectorIF as FaceDetector
from models import GazePredictorHandler as GazePredictor

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import numpy as np


def average_output(out_dict, prev_dict):
    # smooth gaze
    out_dict['gaze_out'] += prev_dict['gaze_out']
    out_dict['gaze_out'] /= np.linalg.norm(out_dict['gaze_out'])
    if out_dict['verts_eyes'] is not None:
        # smooth eyes
        scale_l = np.linalg.norm(out_dict['verts_eyes']['left']) / np.linalg.norm(prev_dict['verts_eyes']['left'])
        scale_r = np.linalg.norm(out_dict['verts_eyes']['right']) / np.linalg.norm(prev_dict['verts_eyes']['right'])
        out_dict['verts_eyes']['left'] *= (1 + (scale_l - 1) / 2) / scale_l
        out_dict['verts_eyes']['right'] *= (1 + (scale_r - 1) / 2) / scale_r
        out_dict['verts_eyes']['left'][:, :2] += - out_dict['verts_eyes']['left'][out_dict['iris_idxs']][:, :2].mean(axis=0) + out_dict['centers_iris']['left']
        out_dict['verts_eyes']['right'][:, :2] += - out_dict['verts_eyes']['right'][out_dict['iris_idxs']][:, :2].mean(axis=0) + out_dict['centers_iris']['right']
    return out_dict


@Timer(name='Forward', fps=True, pprint=False)
def infer_once(img, detector, predictor, draw, prev_dict=None):
    out_img = None
    out_dict = None
    bboxes, lms5, _ = detector.run(img)

    # Supporting only one person
    if bboxes is not None:
        # sort bboxes and pick largest
        idxs_sorted = sorted(range(len(bboxes)), key=lambda k: bboxes[k][3] - bboxes[k][1])
        lms5 = lms5[idxs_sorted[-1]]
        bboxes = bboxes[idxs_sorted[-1]]
        # run inference
        out_dict = predictor(img, lms5, undo_roll=True)
        # out_dict = predictor(img, lms5)
        # smooth output
        if prev_dict is not None:
            out_dict = average_output(out_dict, prev_dict)
        # draw results
        if draw and out_dict is not None:
            out_img = draw_results(img, lms5, out_dict)
    return out_img, out_dict


def inference(cfg, video_path, draw, smooth):
    detector = FaceDetector(cfg.DETECTOR.THRESHOLD, cfg.DETECTOR.IMAGE_SIZE)
    predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)

    loader = VideoLoader(video_path, cfg.DETECTOR.IMAGE_SIZE, use_letterbox=False)
    save_dir = video_path[:video_path.rfind('.')] + f'_out_{cfg.PREDICTOR.BACKBONE_TYPE}_x{cfg.PREDICTOR.IMAGE_SIZE[0]}_{cfg.PREDICTOR.MODE} '
    saver = VideoSaver(output_dir=save_dir, fps=loader.fps, img_size=loader.vid_size, vid_size=640, save_images=False)
    tq = tqdm.tqdm(loader, file=logger)  # tqdm slows down the inference speed a bit

    prev_dict = None
    pred_gaze_all = []
    for frame_idx, input in tq:
        if input is None:
            break
        # infer one frame
        out_img, out_dict = infer_once(input, detector, predictor, draw, prev_dict)
        if out_img is not None:
            prev_dict = out_dict.copy() if smooth else None
            pred_gaze_all += [out_dict['gaze_out']]
        else:
            prev_dict = None
            pred_gaze_all += [(0., 0., 0.)]
        # report 
        description = '{fwd} {ft:.2f} | {det} {det_res:.2f} | {ep} {pred:.2f}'.format(
            fwd='Inference avg fps:',
            det='Detector avg fps:',
            ep='Eye predictor avg fps:',
            ft=Timer.metrics.avg('ForwardGazePredictor'),
            det_res=Timer.metrics.avg('Detector'),
            pred=Timer.metrics.avg('GazePredictor'))
        tq.set_description_str(description)
        if draw and out_img is not None:
            saver(out_img, frame_idx)
    # export predicted gaze for all frames
    with open(f'{save_dir}/predicted_gaze_vectors.txt', 'w') as f:
        f.writelines([f"{', '.join(str(v) for v in p)}\n" for p in pred_gaze_all])


def parse_args():
    parser = argparse.ArgumentParser(description='Inference Gaze')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    known_args, rest = parser.parse_known_args()
    # update config
    update_config(known_args.cfg)
    parser.add_argument('--video_path', help='Video file to run', default="data/test_videos/ms_30s.mp4", type=str)
    parser.add_argument('--gpu_id', help='id of the gpu to utilize', default=0, type=int)
    parser.add_argument('--no_draw', help='Draw and save the results', action='store_true')
    parser.add_argument('--smooth_predictions', help='Average predictions between consecutive frames', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    exp_save_path = f'log/{cfg.EXP_NAME}'
    logger = get_logger(exp_save_path, save=True, use_tqdm=True)
    # ugly workaround
    Timer.save_path = exp_save_path

    with torch.no_grad():
        inference(cfg=cfg, video_path=args.video_path, draw=not args.no_draw, smooth=args.smooth_predictions)
