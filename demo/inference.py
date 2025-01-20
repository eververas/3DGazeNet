import os
import torch
import argparse
import cv2

from models import FaceDetectorIF as FaceDetector
from models import GazePredictorHandler as GazePredictor
from utils import config as cfg
from utils import draw_results

PATH_BASE = os.path.dirname(os.path.abspath(__file__))

class GazeNetInference:
    def __init__(self, det_thresh=0.5, det_size=640):
        '''
            det_thresh: face detection threshold
            det_size: face detection face size - 640, 224
        '''
        self.cfg = cfg
        self.cfg.PREDICTOR.NUM_LAYERS = 18
        self.cfg.PREDICTOR.BACKBONE_TYPE = 'resnet'
        self.cfg.PREDICTOR.PRETRAINED = PATH_BASE + '/data/checkpoints/res18_x128_all_vfhq_vert.pth'
        self.cfg.PREDICTOR.MODE = 'vertex'
        self.cfg.PREDICTOR.IMAGE_SIZE = [ 128, 128 ]
        self.cfg.PREDICTOR.NUM_POINTS_OUT_EYES = 962

        self.face_detector = FaceDetector(det_thresh, det_size)
        self.gaze_predictor = GazePredictor(self.cfg.PREDICTOR, device=self.cfg.DEVICE)

    def run(self, image, kpt=None, draw=False, undo_roll=True):
        '''
            image: cv image
            kpt: face detection 5 keypoints
            draw: falg to draw gaze result
            undo_roll: flag to undo possible face roll before running 3DGazeNet
            ##
            Return: out gaze result list, out image with drawn gaze
        '''
        # fix image channels
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        # face detection + gaze prediction
        with torch.no_grad():
            # detect face
            if kpt is None:
                bboxs, kpts, _ = self.face_detector.run(image)
            # predict gaze
            out_gaze = []
            for kpt in kpts:
                result = self.gaze_predictor(image, kpt, undo_roll=undo_roll)
                out_gaze.append(result)
        # draw
        if draw:
            out_img = image
            for i in range(len(kpts)):
                out_img = draw_results(out_img, kpts[i], out_gaze[i])
        return out_gaze, out_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Gaze')
    parser.add_argument('--image_path', help='Video file to run', default="data/test_images/image00548_0.jpg", type=str)
    parser.add_argument('--no_draw', help='Draw and save the results', action='store_true')
    parser.add_argument('--draw_detection', help='Draw the result of face detector', action='store_true')
    args = parser.parse_args()

    det_thresh = 0.5
    det_size = 224
    gazenet = GazeNetInference(det_thresh, det_size)

    # load image
    image_path = args.image_path
    image = cv2.imread(image_path)

    # test face detector
    if args.draw_detection:
        _, _, faces = gazenet.face_detector.run(image)
        rimg = gazenet.face_detector.draw_bbox(image, faces) 
        cv2.imwrite("./out_facedet.jpg", rimg)

    # run 3DGazeNet
    draw = not args.no_draw
    out_gaze, out_img = gazenet.run(image=image, draw=draw)
    if draw:
        cv2.imwrite('out_gaze.jpg', out_img)
