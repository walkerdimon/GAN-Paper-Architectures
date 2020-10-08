import numpy as np
import torch
import torch.cuda
from torch.utils.data import DataLoader
import argparse
import random
import pdb 
import os 
from bigan import BiGAN

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=.0002)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--model_name", type=str, default='bi-gan')
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--decay_rate", type=float, default=0.999)
parser.add_argument("--z_dim", type=int, default=32)


def main():

    #initialize the model
    bigan = BiGAN(args)

    #load the data

    #run training
    for epoch in range(args.epochs):
        print(f"Training Epoch{epoch+1}/{args.epochs}")

        bigan.train(data.cuda())


    #save your model

    #results??



if __name__ == '__main__':
    main()