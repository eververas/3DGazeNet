import tqdm
import torch
import torch.backends.cudnn as cudnn
import os
import argparse

from utils import config as cfg, update_config, get_logger, Timer, VideoLoader, VideoSaver, show_result, draw_results
from models import FaceDetectorHandler as FaceDetector
from models import GazePredictorHandler as GazePredictor

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import cv2
import numpy as np


@Timer(name='Forward', fps=True, pprint=False)
def infer_once(img, detector, predictor, draw):
    out_img = None
    out_gaze = None
    detector_scale = detector.dispatch_gpu(img)
    bboxes, lms5 = detector.get_results(detector_scale)

    # sort faces according to size and keep largest.
    # Supporting only one person
    if bboxes is not None:
        idxs_sorted = sorted(range(bboxes.shape[0]), key=lambda k: bboxes[k][3] - bboxes[k][1])
        lms5 = lms5[idxs_sorted[-1]]
        bboxes = bboxes[idxs_sorted[-1]]
        out_gaze = predictor(img, lms5)
        if draw and out_gaze is not None:
            out_img = draw_results(img, bboxes, out_gaze)
    return out_img, out_gaze


def inference(cfg, video_path, draw):
    detector = FaceDetector(cfg.DETECTOR)
    predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)

    loader = VideoLoader(video_path, cfg.DETECTOR.IMAGE_SIZE, use_letterbox=False)
    save_dir = video_path[:video_path.rfind('.')] + f'_out_{cfg.PREDICTOR.BACKBONE_TYPE}_x{cfg.PREDICTOR.IMAGE_SIZE[0]}'
    saver = VideoSaver(output_dir=save_dir, vid_size=(640, 360), fps=loader.fps, save_images=True)
    tq = tqdm.tqdm(loader, file=logger)  # tqdm slows down the inference speed a bit
    for frame_idx, input in tq:
        if input is None:
            break
        out_img, out_gaze = infer_once(input, detector, predictor, draw)
        if out_img is not None:
            description = '{fwd} {ft:.2f} | {det} {det_res:.2f} | {nms} {asgn:.2f} | {ep} {pred:.2f}'.format(
                fwd='Inference avg fps:',
                det='Detector avg fps:',
                nms='NMS Assigner avg fps:',
                ep='Eye predictor avg fps:',
                ft=Timer.metrics.avg('Forward'),
                det_res=Timer.metrics.avg('Detector'),
                asgn=Timer.metrics.avg('Assigner'),
                pred=Timer.metrics.avg('GazePredictor'))
            tq.set_description_str(description)
            if draw:
                saver(out_img, frame_idx)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference Gaze')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    known_args, rest = parser.parse_known_args()
    # update config
    update_config(known_args.cfg)
    parser.add_argument('--video_path', help='Video file to run', default="data/nadal_aus_open_cropped.mp4", type=str)
    parser.add_argument('--gpu_id', help='id of the gpu to utilize', default=0, type=int)
    parser.add_argument('--no_draw', help='Draw and save the results', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    exp_save_path = f'log/{cfg.EXP_NAME}'
    logger = get_logger(exp_save_path, save=True, use_tqdm=True)
    # ugly workaround
    Timer.save_path = exp_save_path

    with torch.no_grad():
        inference(cfg=cfg, video_path=args.video_path, draw=not args.no_draw)
