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

        self.Generator.zero_grad()
        self.Encoder.zero_grad()
        self.Discriminator.zero_grad()


        #get fake z_dim for generator
        
        #get fake data from generator to send through discriminator

        #get real x_dim for encoder

        #get real z_dim from encoder to send through discriminator

        #send real x and z data into discriminator

        #send fake x and z data into discriminator

        #compute discriminator loss

        #compute generator/encoder loss

        #compute discriminator gradiants and backpropogate 

        #compute generator/encoder gradiants and backpropogate