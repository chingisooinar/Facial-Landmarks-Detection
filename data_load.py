#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:22:27 2021

@author: nuvilabs
"""
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import random
import torchvision
import albumentations as A
class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

# flip
class Flip(object):
      
    def __init__(self,p = 0.5):

        self.p = p

    def __call__(self, sample):
        if random.uniform(0, 1) <= self.p:
            image, key_pts = sample['image'], sample['keypoints']

            image_copy = np.copy(image)
            key_pts_copy = np.copy(key_pts)

            # flip image
            image_copy = cv2.flip(image, 1)

            key_pts_copy[:,0] = image.shape[1] - key_pts_copy[:,0] - 1

            return {'image': image_copy, 'keypoints': key_pts_copy}
        else:
            return sample
# tranforms
class Noise(object):
        

    def __call__(self, sample):
        if random.uniform(0, 1) >= 0.5:
            image, key_pts = sample['image'], sample['keypoints']

            image_copy = np.copy(image)
            key_pts_copy = np.copy(key_pts)

            # Noisy image
            dst = np.empty_like(image)
            noise = cv2.randn(dst, (0,0,0), (10,10,10))
            image_copy = cv2.addWeighted(image_copy, 0.5, noise, 0.5, 30)
            image_copy = cv2.GaussianBlur(image_copy, (5, 5), 0)
            return {'image': image_copy, 'keypoints': key_pts_copy}
        else:
            return sample
class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        
    def __init__(self, dataset):
        if dataset == "Kaggle":
            self.sub = 48.
            self.div = 48.
        else:
            self.sub = 100.
            self.div = 50.
        self.dataset = dataset
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        if self.dataset != "Kaggle":
            image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0

        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - self.sub)/self.div


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    args:
        output_size(tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size


    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        # Scale the keypoints
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image' : img, 'keypoints' : key_pts}


class RandomCrop(object):
    """ Crop randomly the image in a sample.
    args:
        output_size (tuple or int): Desired output size. If int, square crop
                                    is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size


    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # Random sampling
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        image = image[top : top + new_h, left : left + new_w]
        key_pts = key_pts - [left, top]

        return {'image' : image, 'keypoints' : key_pts}

class Albu(object):


    def __init__(self, transform = A.Compose([
    A.Rotate(always_apply=False, p=0.3, limit=(-30, 30), interpolation=0, border_mode=1, value=(0, 0, 0), mask_value=None),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.2),
    A.HorizontalFlip(p=0.5)

], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))):
        self.transform = transform


    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        aug_image = np.copy(image)
        dict_out = self.transform(image= aug_image, keypoints=key_pts)
        aug_image = dict_out['image']
        key_pts = dict_out['keypoints']
        return {'image' : aug_image, 'keypoints' : key_pts}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #transform = torchvision.transforms.ToTensor()#(image_copy)
       # print(image.shape)
        #image = transform(image)
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}