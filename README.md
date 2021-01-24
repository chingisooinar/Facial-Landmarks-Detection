# Facial Landmarks Detection

# About

The project is a solution for [Kaggle Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection) problem using CNN. The project can also be easily adapted for [this dataset](https://www.kaggle.com/selfishgene/youtube-faces-with-facial-keypoints). 

![Facial%20Landmarks%20Detection%20da7c23fe22fb4c9282f99df1cfafec93/example_1.png](Facial%20Landmarks%20Detection%20da7c23fe22fb4c9282f99df1cfafec93/example_1.png)

# Introduction

The objective of this task is to predict keypoint positions on face images. This can be used as a building block in several applications, such as:

- tracking faces in images and video
- analysing facial expressions
- detecting dysmorphic facial signs for medical diagnosis
- biometrics / face recognition

Detecting facial keypoints is a very challenging problem.  Facial features vary greatly from one individual to another, and even for a single individual, there is a large amount of variation due to 3D pose, size, position, viewing angle, and illumination conditions. Computer vision research has come a long way in addressing these difficulties, but there remain many opportunities for improvement.

# Dependencies

- numpy
- pandas
- sklearn
- pytorch
- OpenCV

# Approach

you can try 3 approaches for this one:

1. Drop any sample that doesn't contain the full 15 key points, in this approach you simply ignore the first dataset, you will get an even smaller dataset with 2140 samples, eventually after training and submitting, you will get almost 3.0 loss.
2. Fill any missing point with the previous available one, in this approach you will end up with 7000+ samples, but most of the features are filled and not accurate, surprisingly this approach will get almost 2.4 loss which is better than the first one, a reasonable explanation for this result is providing the model with 5000 more samples with 4 accurate keypoints and 11 inaccurate filled keypoints lower the loss a bit.
3. Enhance the 1st approach by using the ignored dataset (1st dataset) to train a separate model to predict only 4 key points. Why would we do that?, Obviously this model (four-keypoints model) will produce more accurate predictions for those specific key points as the training set contains 7000 samples with accurate labels rather than only 2000 samples (notice that those 4 keypoints are just subset of the 15 keypoints). In this case, we have 2 models, fifteen-keypoints model which produces 30-dim vector for each sample, and four_keypoints model 8-dim vector for each sample (which produces more accurate values for certain four key points), then you should replace the predictions of the four-keypoints model with the corresponding predictions of the fifteen-keypoints model. This approach will lower loss to almost 2.1, this simply because we got more accurate predictions for 8 features.

# Quick Start

About files:

[train.py](http://train.py) - the main script

[trainer.py](http://trainer.py) - the training script

validater.py - the validation script

data_load.py - useful augmentations

[submit.py](http://submit.py) - merge predictions of two models

[utils.py](http://utils.py) - load and create datasets, visualize predictions.

# Networks

For this project I tried different kind of networks such as

- NaimishNet proposed by [this paper.](https://arxiv.org/pdf/1710.00977.pdf)
- [VGGFace](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/).
- LeNet5.
- Some architectures presented by top Kaggle solutions.

However, the following worked best for me:

```python
Net2(
(conv_layers): Sequential(
(0): Conv2d(1, 4, kernel_size=(5, 5), stride=(1, 1))
(1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(2): ReLU(inplace=True)
(3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(4): Dropout(p=0.2, inplace=False)
(5): Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1))
(6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(7): ReLU(inplace=True)
(8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(9): Dropout(p=0.2, inplace=False)
(10): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
(11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(12): ReLU(inplace=True)
(13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(14): Dropout(p=0.2, inplace=False)
(15): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
(16): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
(17): ReLU(inplace=True)
(18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(19): Dropout(p=0.2, inplace=False)
)
(fc_layers): Sequential(
(0): Linear(in_features=4096, out_features=1024, bias=True)
(1): ReLU(inplace=True)
(2): Dropout(p=0.2, inplace=False)
(3): Linear(in_features=1024, out_features=256, bias=True)
(4): ReLU(inplace=True)
(5): Dropout(p=0.2, inplace=False)
(6): Linear(in_features=256, out_features=8, bias=True)
)
)
```

All the models can be found in models.py

# Results

![Facial%20Landmarks%20Detection%20da7c23fe22fb4c9282f99df1cfafec93/example_4.png](Facial%20Landmarks%20Detection%20da7c23fe22fb4c9282f99df1cfafec93/example_4.png)

### I achieved a value for SmoothL1Loss of about 1.60

### Achieved Score: 2.98340

# Acknowledgment

The data set for this competition was graciously provided by Dr. Yoshua Bengio of the University of Montreal to Kaggle.