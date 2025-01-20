
# 3DGazeNet Demo

<div align="center">
  <p>
      <img src="assets/demo_vid_screenshot.png" width="45%" style="border-radius:0%" alt="Gaze Model Approach">
  </p>
</div>

This is an inference demo of 3DGazeNet trained on four public gaze datasets, namely [`ETH-XGaze`](https://ait.ethz.ch/xgaze) (80 ids, ~750K images), [`GazeCapture`](https://gazecapture.csail.mit.edu/) (1450 ids, ~2M images), [`Gaze360`](http://gaze360.csail.mit.edu/) (238 ids, ~150K images) and [`MPIIFaceGaze`](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation) (15 ids, ~45K images) and one in-the-wild face dataset [`VFHQ`](https://liangbinxie.github.io/projects/vfhq/).


## Installation
Create a conda environment and install dependencies using the following command:

```
$ conda create -n 3dgazenet python=3.9
$ conda activate 3dgazenet
$ pip install -r requirements.txt
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

<!-- ## Download Model and Data

Download and extract the data directory from [here](https://drive.google.com/file/d/13Xw1Dx49oJ45TztACz_5fwy38cwvXMdD/view?usp=sharing). Place the data folder in the ./demo directory. -->

## Run the Demo
### Videos

To run the demo on an input video, execute the below command:
```
$ python inference_video.py --cfg configs/infer_res34_x128_all_vfhq_vert.yaml 
                            --video_path data/test_videos/ms_30s.mp4 
                            --smooth_predictions
```
- The result frames, video and gaze directions are stored in the `data/test_videos` directory as `video_out_path/frame_idx.png`, `output.mp4` and `predicted_gaze_vectors.txt` respectively.
- By default, the demo tracks the largest face in each frame based on face detection.
- `--no_draw`: disables drawing and exporting result frames and video. This speeds up the overall FPS.
- `--smooth_predictions` enables averaging predictions between consecutive frames. In this way, predictions are presented smoother and more consistent in the output videos.


### Images

To run the demo on an input image, execute the below command:
```
$ python inference.py --image_path data/test_images/img1.jpg
```
- By default, the demo detects all faces in the image and runs the model for each.
- `--no_draw`: disables drawing and exporting result frames and video. This speeds up the overall FPS.
- `--draw_detection`: returns an image with the face detection boxes.


or from outside this directory, use model inference in the following way:
```
import cv2

import sys
sys.path.append('...') # path to 3DGazeNet demo directory
from inference import GazeNetInference

# init 3DGazeNet and face detector
det_thresh = 0.5
det_size = 224
gazenet = GazeNetInference(det_thresh, det_size)

# load image
image_path = '...' 
image = cv2.imread(image_path)

# run 3DGazeNet
out_gaze, out_img = gazenet.run(image=image, draw=True)

print(out_gaze)
cv2.imwrite('out_gaze.jpg', out_img)

```

## Performance

### - Gaze Error in degrees ($^o$) on public gaze datasets

The performance of the provided model on the test sets of the four public gaze datasets is presented in the Table below:

|              | ETH-XGaze | GazeCapture | Gaze360 | MPIIFaceGaze | 
| ------------ | :-------: | :---------: | :-----: | :----------: |
| Error ($^o$) | 4.2       | 3.3         | 8.8     | 4.3          |

### - Model Info

|   Backbone   | Input Size<br><sup>(pixels) | ModelParams<br><sup>(Million) | Size<br><sup>(MB) | FLOPs<br><sup>(GB)<br> | FPS-RTX4090<br><sup>(Gaze Model)<br> |FPS-RTX4090<br><sup>(Overall, w/o drawing)<br> | 
| :----------: | :-------------------------: | :--------------------------: | :---------------: | :--------------------: | :--------------------: | :--------------------: |
| Resnet18     | 128x128                     | 30.2                         | 330               | 5.1                    | ~726                   |  ~68                   |

