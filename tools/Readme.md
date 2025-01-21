## Prepare Datasets 

### ETH-XGaze
1\. Download the 448*448 pixels version of the datset from the official source [here](https://ait.ethz.ch/xgaze). Place the dataset in the `datasets` folder in the root of this repo.

2\. Fit 3D eyes on images using the following command. This will export a `.pkl` data file in the dataset's folder which is used for training.
```
cd tools
python xgaze_preprocess.py
```

3\. To visualize the 3D eye fittings run the notebook in `notebooks/xgaze_view_dataset.ipynb`.

### Gaze360

1\. Download the datset from the official source [here](http://gaze360.csail.mit.edu/). Place the dataset in the `datasets` folder in the root of this repo.

2\. Fit 3D eyes on images using the following command. This will export a `.pkl` data file in the dataset's folder which is used for training.
```
cd tools
python gaze360_preprocess.py
```

3\. To visualize the 3D eye fittings run the notebook in `notebooks/gaze360_view_dataset.ipynb`.

### GazeCapture

1\. Download the datset from the official source [here](https://gazecapture.csail.mit.edu/). Place the dataset in the `datasets` folder in the root of this repo.

### MPIIFaceGaze

1\. Download the aligned dataset from the official source [here](https://www.perceptualui.org/research/datasets/MPIIFaceGaze/). Place the dataset in the `datasets` folder in the root of this repo.

2\. Fit 3D eyes on images using the following command. This will export a `.pkl` data file in the dataset's folder which is used for training.
```
cd tools
python mpiiface_preprocess.py
```

3\. To visualize the 3D eye fittings run the notebook in `notebooks/mpiiface_view_dataset.ipynb`.
