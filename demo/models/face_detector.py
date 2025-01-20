import numpy as np
import torch
from utils import Timer

class FaceDetectorIF:
    def __init__(self, det_thresh=0.5, det_size=640):
        from insightface.app import FaceAnalysis
        self.model = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(det_size, det_size)) # 640, 224

    @Timer(name='Detector', fps=True, pprint=False)
    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list, kpt5 list
        '''
        faces = self.model.get(image)
        if len(faces) == 0:
            return [0], None, None
        # bbox = faces[0].bbox.astype(int)
        # kpt = faces[0].kps.astype(int)
        bbox = [face.bbox.astype(int) for face in faces]
        kpt = [face.kps.astype(int) for face in faces]
        return bbox, kpt, faces
    
    def draw_bbox(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 4:
                        color = (255, 0, 0)
                    if l == 1 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
        return dimg
        



