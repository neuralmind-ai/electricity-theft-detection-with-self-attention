import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(2, 64, kernel_size=3,padding=1)),
            ('prelu1', nn.PReLU()),

            ('conv2', nn.Conv2d(64, 64, kernel_size=3,padding=1)),
            ('prelu2', nn.PReLU()),
            ('drop2', nn.Dropout(p=0.4)),

            ('conv3', nn.Conv2d(64, 32, kernel_size=3,padding=2, dilation=2, stride=2)),
            ('prelu3', nn.PReLU()),
            ('drop3', nn.Dropout(p=0.7)),
        ]))
        
        self.dense_layer = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(32*74*4, 280)),
            ('prelu1',  nn.PReLU()),
            ('drop1',  nn.Dropout(p=0.7)),

            ('dense2', nn.Linear(280, 140)),
            ('prelu2', nn.PReLU()),
            ('dropout2', nn.Dropout(p=0.6)), 

           
           ('dense3', nn.Linear(140,2)),    
        ]))
        
    def forward(self, x):
        x = self.conv_layer(x)
        #print(x.shape)
        x = x.view(-1, 32*74*4) 
        x = self.dense_layer(x)
        return x

