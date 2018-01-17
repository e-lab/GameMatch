import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable as V
from math import ceil
import time
import copy
from torchvision.utils import save_image
import os
import shutil
import argparse

class Bottleneck(nn.Module):
    def __init__(self, channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels//2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels//2)
        self.conv2 = nn.Conv2d(channels//2, channels//2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels//2)
        self.conv3 = nn.Conv2d(channels//2, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(channels),)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Cortexnet_a2c(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Cortexnet_a2c,self).__init__()
        self.image_sizes=[input_size]
        for layer in range(4):
            self.image_sizes.append(tuple(s//2 for s in self.image_sizes[layer]))
        self.conv1 = nn.Conv2d(3,32, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.fc1 = nn.Linear(256*8*10,1024)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024,num_classes)
        #self.value = nn.Linear(1024,1)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(64, 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

        self.ebn1 = nn.BatchNorm2d(32)
        self.ebn2 = nn.BatchNorm2d(64)
        self.ebn3 = nn.BatchNorm2d(128)
        self.ebn4 = nn.BatchNorm2d(256)
        self.dbn4 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)

    def forward(self, x, state):
        if state is None:
            state=[None]*3
        residuals=[]
        x=self.relu(self.conv1(x))
        residuals.append(x)
        if state[0] is None:
            state[0]=V(x.data.clone().zero_())
        x=torch.cat((x,state[0]), 1)
        x=self.relu(self.conv2(x))
        residuals.append(x)
        if state[1] is None:
            state[1]=V(x.data.clone().zero_())
        x=torch.cat((x,state[1]), 1)
        x=self.relu(self.conv3(x))
        residuals.append(x)
        if state[2] is None:
            state[2]=V(x.data.clone().zero_())
        x=torch.cat((x,state[2]), 1)
        x=self.relu(self.conv4(x))

        y=x.view(-1,256*8*10)
        y=self.dropout1(self.relu(self.fc1(y)))
        v=self.value(y)
        y=self.fc2(y)

        x=self.relu(self.deconv4(x, self.image_sizes[3]))
        state[2]=x
        x=torch.cat((x,residuals[2]), 1)
        x=self.relu(self.deconv3(x, self.image_sizes[2]))
        state[1]=x
        x=torch.cat((x,residuals[1]), 1)
        x=self.relu(self.deconv2(x, self.image_sizes[1]))
        state[0]=x
        x=torch.cat((x,residuals[0]), 1)
        x=self.deconv1(x, self.image_sizes[0])
        x=self.tanh(x)

        return (x,y,v,state)
