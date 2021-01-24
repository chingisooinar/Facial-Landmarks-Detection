#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:31:10 2021

@author: nuvilabs
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import os
from torch.utils.data import Dataset
class KagggleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, y, transform=None):
        """
        Args:
            X :images, y: annotations 
            
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        image = self.X[idx]
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.y[idx]
       # print(key_pts)
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
def load_KagggleDataset(test=False, complete_points=True,split=False,train_8=False,train_30=False):
    """
    Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Important that the files are in a `data` directory
    """  
    FTRAIN = 'kaggle/training.csv'
    FTEST = 'kaggle/test.csv'
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load dataframes

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    if not split:
        if complete_points:
            df = df.dropna()  # drop all rows that have missing values in them
        else:
            df.fillna(df.describe().T['50%'], inplace=True)
    else:
        if train_8:
            feature_8 = ['left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x','right_eye_center_y',
            'nose_tip_x', 'nose_tip_y',
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y', 'Image']

            df = df[feature_8].dropna()
        else:
            df = df.dropna()

    X = np.vstack(df['Image'].values)  # get image
    X = X.astype(np.uint8)
    X = X.reshape(-1,96,96,1)

    if not test:  # only FTRAIN has target columns
        y = df[df.columns[:-1]].values
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y
def net_sample_output(net, test_loader, device, dataset):
    if dataset == "Kaggle":
        norm = 15
    else:
        norm = 68
    net.to(device)
    net.eval()
    with torch.set_grad_enabled(False):
    # iterate through the test dataset
        for i, sample in enumerate(test_loader):

            # get sample data: images and ground truth keypoints
            images = sample['image']
            key_pts = sample['keypoints']
        
            # convert images to FloatTensors
            images = images.float()
            images = images.to(device)
            images = Variable(images, volatile=True)
            key_pts = Variable(key_pts, volatile=True)
            # forward pass to get net output
            output_pts = net(images)

            # reshape to batch_size x 68 x 2 pts
            output_pts = output_pts.view(output_pts.size()[0], norm, -1)

            # break after first image is tested
            if i == 1:
                return images, output_pts, key_pts
def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    #print(image)
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0] , gt_pts[:, 1], s=20, marker='.', c='g')
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:s
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
def visualize_output(test_images, test_outputs, gt_pts=None, dataset ="Kaggle",start_idx = 0 , batch_size=10):
    if dataset == "Kaggle":
        sub = 48.
        div = 48.
    else:
        sub = 100.
        div = 50.
    for i in range(start_idx,start_idx + batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the image data
        image = test_images[i].data.cpu()   # get the image from it's Variable wrapper

        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image
        image = image*255.0
        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data.cpu()
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*div+sub
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*div+sub
        
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        plt.axis('off')

    plt.show()