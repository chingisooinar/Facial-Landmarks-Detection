#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 15:13:36 2021

@author: nuvilabs
"""


# import the usual resources
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
from utils import show_all_keypoints, visualize_output,net_sample_output, load_KagggleDataset, KagggleDataset
# one example conv layer has been provided for you
from models import Net, VggFace,NaimishNet, Net2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from data_load import FacialKeypointsDataset
from data_load import Normalize, ToTensor
from trainer import train_net
from validater import val_net
idlookup_file = './kaggle/IdLookupTable.csv'
def plot_face_pts(img, pts):
    plt.imshow(img[:,:,0], cmap='gray')
    for i in range(1,31,2):
        plt.plot(pts[i-1], pts[i], 'b.')

 #load models   
net_8 = Net2(8)
net_8.load_state_dict(torch.load('./saved_models/Kaggle_Custom_1.68410162627697.pt'))
net_30 =Net2(30)
net_30.load_state_dict(torch.load('./saved_models/Kaggle_Custom_1.6080801971256733.pt'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


data_transform = transforms.Compose([Normalize("Kaggle"),ToTensor()])

# testing that you've defined a transform
assert(data_transform is not None), 'Define a data_transform'
    


X, y = load_KagggleDataset(test=True)

y = np.zeros((X.shape[0],30))

test_dataset = KagggleDataset(X, y, data_transform)
batch_size =32

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=False, 
                          num_workers=0)
#get predictions
all_preds = []
for net in [net_8,net_30]:
    net.to(device)
    net.eval()
    test_preds = []      
    with torch.set_grad_enabled(False):
        for i,data in enumerate(test_loader):
            images = data['image']
            # flatten pts
            # convert variables to floats for regression loss
            images = images.float()
            images = images.to(device)
            prediction = net(images)
            prediction = prediction * 48. + 48.
            test_preds.extend(prediction.cpu().numpy())
    all_preds.append(test_preds)

y_hat_30 = all_preds[1]
y_hat_8 = all_preds[0]
y_hat_8 = np.asarray(y_hat_8)
y_hat_30 = np.asarray(y_hat_30)
feature_8_ind = [0, 1, 2, 3, 20, 21, 28, 29]
#Merge 2 prediction from y_hat_30 and y_hat_8.
for i in range(8):
    print('Copy "{}" feature column from y_hat_8 --> y_hat_30'.format(feature_8_ind[i]))
    y_hat_30[:,feature_8_ind[i]] += y_hat_8[:,i]
    y_hat_30[:,feature_8_ind[i]] /= 2
#plot
fig = plt.figure(figsize=(10, 7))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i, f in enumerate(range(200,216)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_face_pts(X[f], y_hat_30[f])

plt.show()