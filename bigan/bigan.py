import numpy as np
import torch
import torch.cuda
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import random
import pdb 
import os 

from networks import Discriminator, Generator, Encoder
from utils import *

class BiGAN(object):
    def __init__(self, args):

        self.z_dim = args.z_dim
        self.decay_rate = args.decay_rate
        self.learning_rate = args.learning_rate
        self.model_name = args.model_name
        self.batch_size = args.batch_size

        #initialize networks
        self.Generator = Generator(self.z_dim).cuda()
        self.Encoder = Encoder(self.z_dim).cuda()
        self.Discriminator = Discriminator().cuda()

        #set optimizers for all networks
        self.optimizer_G_E = torch.optim.Adam(list(self.Generator.parameters()) + list(self.Encoder.parameters()),
                                                                         lr=self.learning_rate, betas=(0.5, 0.999))

        self.optimizer_D = torch.optim.Adam(self.Discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        #initialize network weights
        self.Generator.apply(weights_init)
        self.Encoder.apply(weights_init)
        self.Discriminator.apply(weights_init)

    def train(self, data):

        self.Generator.train()
        self.Encoder.train()
        self.Discriminator.train()

        self.optimizer_G_E.zero_grad()
        self.optimizer_D.zero_grad()


        #get fake z_data for generator
        self.z_fake = torch.randn((self.batch_size, self.z_dim))
        
        #send fake z_data through generator to get fake x_data
        self.x_fake = self.Generator(self.z_fake.detach())

        #send real data through encoder to get real z_data
        self.z_real = self.Encoder(data)

        #send real x and z data into discriminator
        self.out_real = self.Discriminator(data, z_real.detach())

        #send fake x and z data into discriminator
        self.out_fake = self.Discriminator(x_fake.detach(), z_fake.detach())

        #compute discriminator loss
        self.D_loss = nn.BCELoss()

        #compute generator/encoder loss
        self.G_E_loss = nn.BCELoss()

        #compute discriminator gradiants and backpropogate 
        self.D_loss.backward()
        self.optimizer_D.step()

        #compute generator/encoder gradiants and backpropogate
        self.G_E_loss.backward()
        self.optimizer_G_E.step()