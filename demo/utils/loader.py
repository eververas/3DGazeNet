import cv2
from .utils import letterbox


class VideoLoader:
    def __init__(self, video_path, img_size, use_letterbox=False):
        self.img_size = img_size
        self.frame = 0
        self.cap = cv2.VideoCapture(video_path)
        self.use_letterbox = use_letterbox
        assert self.cap.isOpened(), 'Failed to load %s' % video_path
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.vid_size = (width, height)

    def __iter__(self):
        return self

    def __len__(self):
        return self.frames

    def resize(self, img0):
        shape = img0.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        # only scale down, do not scale up (for better test mAP)
        r = min(min(self.img_size / self.vid_size[0], self.img_size / self.vid_size[1]), 1.0)
        new_unpad = int(round(self.vid_size[0] * r)), int(round(self.vid_size[1] * r))
        img = cv2.resize(img0, new_unpad, interpolation=cv2.INTER_LINEAR)
        return img

    def __next__(self):
        # Read video
        ret_val, img0 = self.cap.read()
        frame_idx = int(self.cap.get(1))  # cv2.CV_CAP_PROP_POS_FRAMES)

        self.frame += 1
        if img0 is None or not ret_val:
            self.cap.release()
            raise StopIteration
        if self.use_letterbox:
            img = letterbox(img0, self.img_size)[0]
        else:
            img = self.resize(img0)
        return frame_idx, img
