import numpy as np 
import torch 
from torch import autograd 
import torch.nn as nn 
import os 

#Xavier weight initialization
def weights_init(m):
    
    if isinstance(m, nn.ConvTranspose2d):
        if m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv2d):
        if m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    if isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)