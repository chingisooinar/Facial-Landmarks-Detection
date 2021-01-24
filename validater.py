#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:02:16 2021

@author: nuvilabs
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def val_net(epoch, n_epochs, net, criterion, optimizer, device, test_loader, sub, div):
    net.eval()      
    val_loss=0.0
    print(device)
    with torch.set_grad_enabled(False):
        for i,data in enumerate(test_loader):
            images = data['image']
            key_pts = data['keypoints']
            optimizer.zero_grad()

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)
    
            # convert variables to floats for regression loss
            key_pts = key_pts.float()
            images = images.float()
            images = images.to(device)
            key_pts = key_pts.to(device)
    
            prediction = net(images)
            #print(prediction.shape,angles.shape)
            loss = torch.sqrt(criterion(prediction * div + sub, key_pts * div + sub)) if criterion(prediction, key_pts) == nn.MSELoss()(prediction, key_pts) else criterion(prediction * div + sub, key_pts * div + sub)
    
            val_loss+=loss.data.item()
            
            print(f'Test Loss at {i}: {val_loss / (i + 1)}')
    return val_loss / (i + 1)