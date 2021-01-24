#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 10:45:03 2021

@author: nuvilabs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import numpy as np

class Net(nn.Module):

    def __init__(self,n):
        super(Net, self).__init__()
        self.build_model(n)
        
    def build_model(self,n):
                                                            
               
        self.conv_layers = nn.Sequential(
          nn.Conv2d(1,32,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(32),
          
          nn.Conv2d(32,32,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(32),
          nn.MaxPool2d(2),
          
          nn.Conv2d(32,64,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(64),
          
          nn.Conv2d(64,64,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(2),
          
          nn.Conv2d(64,96,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(96),
          
          nn.Conv2d(96,96,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(96),
          nn.MaxPool2d(2),
          
          nn.Conv2d(96,128,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(128),
          
          nn.Conv2d(128,128,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(128),
          nn.MaxPool2d(2),
          
          nn.Conv2d(128,256,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(256),
          
          nn.Conv2d(256,256,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(256),
          nn.MaxPool2d(2),

          nn.Conv2d(256,512,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(512),
          nn.Dropout(0.),
          
          nn.Conv2d(512,512,kernel_size=(3,3), padding=1),
          nn.LeakyReLU(negative_slope=0.1,inplace=True),
          nn.BatchNorm2d(512),
          nn.Dropout(0.)
          #nn.MaxPool2d(2),          

        )
        
     
        self.fc_layers = nn.Sequential(
          nn.Linear(4608, 512),
          nn.ReLU(inplace=True),
          nn.Dropout(0.5),
          nn.Linear(512,n),
 
        )
        
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.reshape(out.shape[0],-1)
        #print(out.shape)
        out = self.fc_layers(out)
        
        return out

        
class VggFace(nn.Module):
    def __init__(self, classes=2622):
        """VGGFace model.
        Face recognition network.  It takes as input a Bx3x224x224
        batch of face images and gives as output a BxC score vector
        (C is the number of identities).
        Input images need to be scaled in the 0-1 range and then 
        normalized with respect to the mean RGB used during training.
        Args:
            classes (int): number of identities recognized by the
            network
        """
        super(VggFace, self).__init__()
        self.conv1 = ConvBlock(1, 64, 64)
        self.conv2 = ConvBlock(64, 128, 128)
        self.conv3 = ConvBlock(128, 256, 256, 256)
        self.conv4 = ConvBlock(256, 512, 512, 512)
        self.conv5 = ConvBlock(512, 512, 512, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3 * 3 * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
class ConvBlock(nn.Module):
    def __init__(self, *params):
        super(ConvBlock,self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_features, out_features, 3, 1, 1)
            for in_features, out_features in zip(params[:-1],params[1:])])
        
        
    def forward(self, x):
    
        for layer in self.convs:
            x = F.relu(layer(x))
        x = F.max_pool2d(x, 2, 2, 0, ceil_mode=True)
        
        return x
class NaimishNet(nn.Module):
    
    def __init__(self,n):
        super(NaimishNet, self).__init__()

        """
        NaimishNet has layers below:
        Layer Num  | Number of Filters | Filter Shape
         ---------------------------------------------
        1          |        32         |    (4,4)
        2          |        64         |    (3,3)
        3          |        128        |    (2,2)
        4          |        256        |    (1,1)
        ---------------------------------------------
        Activation : ELU(Exponential Linear Unit)
        MaxxPool : 4x(2,2)
        """

        self.max_pool = nn.MaxPool2d(2, 2)
    
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.drop1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.drop2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.drop3 = nn.Dropout(0.3)
        
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.drop4 = nn.Dropout(0.4)
        
        self.dense1 = nn.Linear(6400, 1000)
        self.drop5 = nn.Dropout(0.5)
        
        self.dense2 = nn.Linear( 1000,  1000)
        self.drop6 = nn.Dropout(0.6)
        
        self.dense3 = nn.Linear( 1000, n)


    def forward(self, x):
        #print(x.shape)
        x = self.max_pool(F.elu(self.conv1(x)))
        x = self.drop1(x)

        x = self.max_pool(F.elu(self.conv2(x)))
        x = self.drop2(x)

        x = self.max_pool(F.elu(self.conv3(x)))
        x = self.drop3(x)

        x = self.max_pool(F.elu(self.conv4(x)))
        x = self.drop4(x)

        # Flatten layer
        x = x.view(x.size(0), -1)
       # print(x.shape)
        x = F.elu(self.dense1(x))
        x = self.drop5(x)

        x = F.relu(self.dense2(x))
        x = self.drop6(x)

        x = self.dense3(x)

        return x    
    
class Net2(nn.Module):

    def __init__(self,n):
        super(Net2, self).__init__()
        self.build_model(n)
        
    def build_model(self,n):
                                                            
               
        self.conv_layers = nn.Sequential(
          nn.Conv2d(1,4,kernel_size=5),
          nn.BatchNorm2d(4),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),
          nn.Dropout(0.2),
          
          nn.Conv2d(4,64,kernel_size=3),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),
          nn.Dropout(0.2),   
          
          nn.Conv2d(64,128,kernel_size=3),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),
          nn.Dropout(0.2),
          
          nn.Conv2d(128,256,kernel_size=3),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),
          nn.Dropout(0.2)
          
        )
        
     
        self.fc_layers = nn.Sequential(
          nn.Linear(4096, 1024),
          nn.ReLU(inplace=True),
          nn.Dropout(0.2),

          nn.Linear(1024, 256),
          nn.ReLU(inplace=True),
          nn.Dropout(0.2),
          
          nn.Linear(256, n)
       
 
        )
        
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.reshape(out.shape[0],-1)
        #print(out.shape)
        out = self.fc_layers(out)
        
        return out   

class LeNet5(nn.Module): 
    
    def __init__(self,n):
        super(LeNet5, self).__init__()
        # Convolution Block
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 6 input image channels, 16 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully connected Block
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 21 * 21, 120)  # 21*21 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n)

    def forward(self, x):
        
        # The image size of our dataset is 96x96, then
        # Input size: 96x96   
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        # Output from layer conv1: 6 layers of 96x96 size (96-5+1 = 92)
        # Output after Max Pooling window (2,2): (92-2+2)/2 = 46
        x = F.max_pool2d(torch.sigmoid(self.conv1(x)), 2, 2) 
        
        # Input size for the next layer: 46x46
        # 6 input image channel, 16 output channels, 5x5 square convolution kernel
        # Output from layer conv1: 16 layers of 46x46 size (46-5+1 = 42)
        # Output after Max Pooling window (2,2): (42-2+2)/2 = 21
        x = F.max_pool2d(torch.sigmoid(self.conv2(x)), 2, 2) 
        
        # Input size for the fully connected stage: 16 layers of 21x21 size
        x = x.view(x.shape[0], -1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
