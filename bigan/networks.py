# Network Architecture for Bigan: Generator, Discriminator, and Encoder

import torch
import torch.nn as nn 
import numpy as np 
import torch.nn.functional as F 
from torch.nn import Parameter as P 
import pdb 


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()

        self.Activation = nn.ReLU(inplace=True)

        self.dense_net = nn.Linear(z_dim, 256*4*4)

        self.normalization = nn.BatchNorm2d()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2, bias=True) #4x4 --> 8x8

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True) #8x8 --> 16x16

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2, bias=True) #16x16 --> 32x32


    def forward(self, z):

        x = self.dense_net(z)
        x = self.Activation(x)
        
        x = x.view(x.size(0), 256, 4, 4)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv1(x)
        x = self.Activation(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.normalization(128)

        x = self.conv2(x)
        x = self.Activation(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.normalization(64)

        x = self.conv3(x)
        x = self.Activation(x)

        x = torch.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.Activation = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        self.normalization = nn.BatchNorm2d()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, bias=True)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True)

        self.dense_net = nn.Linear(128*4*4, 1)

    
    def forward(self, x):

        d = self.conv1(x)
        d = self.Activation(d)
        d = self.normalization(d)

        d = self.conv2(d)
        d = self.Activation(d)
        d = self.normalization(d)

        d = self.conv3(d)
        d = self.Activation(d)

        d = d.view(d.size(0), 128*4*4)
        d = self.dense_net(d)


        return d


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Enocder, self).__init__()

        self.Activation = nn.LeakyReLU(inplace=True, negative_slope=0.2)

        self.normalization = nn.BatchNorm2d()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, bias=True) #32x32 --> 14x14
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True) #14x14 --> 7x7

        self.conv3 = nn.conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True) #7x7 --> 4x4

        self.dense_net = nn.Linear(256*4*4, z_dim)

    def forward(self, x):

        z = self.conv1(x)
        z = self.Activation(z)
        z = self.normalization(64)

        z = self.conv2(z)
        z = self.Activation(z)
        z = self.normalization(128)

        z = self.conv3(z)
        z = self.Activation(z)

        z = z.view(z.size(0), 256*4*4)
        z = self.dense_net(z)

        z = torch.tanh(z)

        return z



