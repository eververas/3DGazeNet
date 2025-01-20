import os
import cv2


class VideoSaver:
    def __init__(self, output_dir, fps, img_size, vid_size, save_images=False):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs(output_dir, exist_ok=True)
        output_fname = os.path.join(output_dir, 'output.mp4')
        shape = img_size
        r = min(min(vid_size / shape[0], vid_size / shape[1]), 1.0)
        width, height = int(round(shape[0] * r)), int(round(shape[1] * r))
        self.saver = cv2.VideoWriter(output_fname, fourcc, float(fps), (width, height))
        self.fps = fps
        self.vid_size = vid_size
        self.save_images = save_images
        self.output_dir = output_dir

    def __iter__(self):
        return self

    def resize(self, img0):
        shape = img0.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        # only scale down, do not scale up (for better test mAP)
        r = min(min(self.vid_size / shape[0], self.vid_size / shape[1]), 1.0)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        img = cv2.resize(img0, new_unpad, interpolation=cv2.INTER_LINEAR)
        return img

    def __call__(self, frame, frame_idx):
        if self.save_images:
            cv2.imwrite(f'{self.output_dir}/{frame_idx}.jpg', frame)
        frame = self.resize(frame)
        self.saver.write(frame)


if __name__ == '__main__':
    import glob

    dir = './'
    list_images = glob.glob(dir + '/*.png')
    list_images = sorted(list_images, key=lambda x: int(x.split('/')[-1][2:-4]))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fname = os.path.join(dir, 'a_output.mp4')
    saver = cv2.VideoWriter(output_fname, fourcc, 30., (640, 360))
    for img_path in list_images:
        saver.write(cv2.imread(img_path))

    saver.release()
