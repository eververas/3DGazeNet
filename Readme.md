# 3DGazeNet: Generalizing Gaze Estimation with Weak-Supervision from Synthetic Views ([arxiv](https://arxiv.org/abs/2212.02997))

https://github.com/Vagver/dense3Deyes/assets/25174551/4de4fb76-9577-4209-ba07-779356230131

## Installation

To create a conda environment with the required dependences run the command: 

```
$ conda env create --file env_requirements.yaml
$ conda activate 3DGazeNet
```

## Download models

Download the data directory contatining pre-trained gaze estimation models from [here](sdsdsd). Extract and place the data folder in the root directory of this repo.

## Inference

To run inference on a set of images follow the steps below. A set of example images are given in the `data/example_images` directory.

1\. Pre-process the set of images. This step performs face detection and exports a `.pkl` file in the path defined by `--output_dir`, containing pre-processing data. For data pre-processing run the following command:

```
$ cd tools
$ python preprocess_inference.py --image_base_dir ../data/example_images \
                                 --output_dir ../output/preprocessing \
                                 --gpu_id 0 --n_procs 5
```

2\. Run inference on the set of images. This step outputs gaze estimation and 3D eye reconstruction results in a `.pkl` file in the `inference_results` directory. with the following command:

```
$ python inference.py --cfg configs/inference/inference.yaml \
                      --inference_data_file 'output/preprocessing/data_face68.pkl' \
                      --inference_dataset_dir 'data/example_images/' \
                      --checkpoint output/models/singleview/vertex/ALL/test_0/checkpoint.pth \
                      --skip_optimizer
```

3\. To inspect the gaze tracking results run the jupyter notebook in `notebooks/view-inference_results.ipynb`


