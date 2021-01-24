#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:23:10 2021

@author: nuvilabs
"""
# import the usual resources
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils import show_all_keypoints, visualize_output,net_sample_output, load_KagggleDataset, KagggleDataset
# one example conv layer has been provided for you
from models import Net, VggFace,NaimishNet, Net2, LeNet5
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor,Flip,Noise,Albu
from trainer import train_net
from validater import val_net
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epoch",default=80, type=int, help="number of epochs")
ap.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
ap.add_argument("-d", "--dataset", default="Kaggle", type=str, help="Kaggle or Udacity")
ap.add_argument("-m", "--model", default="Custom", type=str, help="Kaggle or Udacity")
ap.add_argument("-l", "--loss", default="MSE", type=str, help="Kaggle or Udacity")
args = vars(ap.parse_args())
if args['model'] == "NaimishNet":
    net = NaimishNet(8)
elif args['model'] == "VggFace":
    net = VggFace(8)    
elif args['model'] == "Custom":
    net = Net2(8)
else:
    net = LeNet5(8)
model_name = args['model']
print(net)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# order matters! i.e. rescaling should come before a smaller crop
train_transform = transforms.Compose([Rescale(110),RandomCrop(96),Albu(),Normalize(args["dataset"]),ToTensor()])
test_transform = transforms.Compose([Normalize(args["dataset"]),ToTensor()])

# testing that you've defined a transform
assert(train_transform is not None and test_transform is not None), 'Define a data_transform'
    
# create the transformed dataset

if args["dataset"] == "Kaggle":
    X, y = load_KagggleDataset(split=True,train_30=True)
    X_test, y_test = X[:300], y[:300]
    X_train, y_train = X[300:], y[300:]
    transformed_dataset = KagggleDataset(X_train, y_train, train_transform)
    test_dataset = KagggleDataset(X_test, y_test, test_transform)
    sub, div = 48., 48.
else:
    transformed_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                             root_dir='./data/training/',
                                         transform=train_transform)
    test_dataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                             root_dir='./data/test/',
                                            transform = test_transform)
    sub, div = 100.,50.
print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].shape, sample['keypoints'].size())
# load training data in batches
batch_size = 32
train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)
# load test data in batches
batch_size =32

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=False, 
                          num_workers=0)


if args['loss'] == "MSE":
    criterion = nn.MSELoss()
else:
    criterion = nn.SmoothL1Loss()
net.to(device)

optimizer =torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=5e-4) #define optimizer
val_net(0, 0, net, criterion, optimizer, device, test_loader, sub, div)
n_epochs = 600
min_loss = float('inf')
for epoch in range(n_epochs):
    print("=====Training=======")
    train_net(epoch, n_epochs, net, criterion, optimizer, device, train_loader, sub, div)
    print("=====Validation=======")
    loss = val_net(epoch, n_epochs, net, criterion, optimizer, device, test_loader, sub, div)
    if loss < min_loss:
        print("=====Saving=======")
        model_dir = './saved_models/'
        name =  args["dataset"]+'_'+model_name+'_'+str(loss)+'.pt'
        min_loss = loss
        # after training, save your model parameters in the dir 'saved_models'
        torch.save(net.state_dict(), model_dir+name)