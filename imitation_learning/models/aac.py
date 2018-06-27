#
# aac.py, doom-net
#
# Created by Andrey Kolishchak on 01/21/17.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        in_channels = 3
        button_num = 32
        
        self.screen_feature_num = 256
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        self.screen_features1 = nn.Linear(512 * 4, self.screen_feature_num)
        #self.screen_features1 = nn.Linear(128 * 6 * 9, self.screen_feature_num)
        #self.screen_features1 = nn.Linear(64 * 14 * 19, self.screen_feature_num)

        self.batch_norm = nn.BatchNorm1d(self.screen_feature_num)

        layer1_size = 128
        self.action1 = nn.Linear(self.screen_feature_num, layer1_size)
        self.action2 = nn.Linear(layer1_size, button_num)


    def forward(self, screen):
        # cnn
        screen_features = F.selu(self.conv1(screen))
        screen_features = F.selu(self.conv2(screen_features))
        screen_features = F.selu(self.conv3(screen_features))
        screen_features = F.selu(self.conv4(screen_features))
        screen_features = F.selu(self.conv5(screen_features))
        screen_features = F.selu(self.conv6(screen_features))
        screen_features = screen_features.view(screen_features.size(0), -1)

        # features
        input = self.screen_features1(screen_features)
        input = self.batch_norm(input)
        input = F.selu(input)

        # action
        action = F.selu(self.action1(input))
        #action = torch.cat([action, )
        action = self.action2(action)

        return action
