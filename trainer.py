#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:57:08 2021

@author: nuvilabs
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
def train_net(epoch, n_epochs, net, criterion, optimizer, device, train_loader, sub, div):

    # prepare the net for training
    net.train()

    print(device)
    running_loss = 0.0

    # train on batches of data, assumes you already have train_loader
    for batch_i, data in enumerate(train_loader):
        # get the input images and their corresponding labels
        images = data['image']
        key_pts = data['keypoints']
        # zero the parameter (weight) gradients
        optimizer.zero_grad()
        
        # flatten pts
        key_pts = key_pts.view(key_pts.size(0), -1)

        # convert variables to floats for regression loss
        key_pts = key_pts.float()
        images = images.float()
        images = images.to(device)
        key_pts = key_pts.to(device)
        images = Variable(images)
        key_pts = Variable(key_pts)
        # forward pass to get outputs
        output_pts = net(images)

        # calculate the loss between predicted and target keypoints
        loss = torch.sqrt(criterion(output_pts* div + sub , key_pts * div + sub)) if criterion(output_pts, key_pts) == nn.MSELoss()(output_pts, key_pts) else criterion(output_pts * div + sub, key_pts * div + sub)
        
        # backward pass to calculate the weight gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # print loss statistics
        running_loss += loss.item()
        if batch_i % 10 == 9:    # print every 10 batches
            print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
            running_loss = 0.0

    #print('Finished Training')