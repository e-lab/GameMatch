# E. Culurciello
# November 2018
# experiments with predictive networks in RL
#
# inspired by: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#
# DEMO code from a pre-trained network

import gym
import math
import random
import numpy as np
from itertools import count
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from termcolor import colored

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import sys

game = 'Breakout-v0'
env = gym.make(game)
print(colored('\nPlaying game:', 'green'), game)
model_file_name = sys.argv[1]
print('Demo using pre-trained network:', model_file_name)
print('Usage: python3 demo_trained.py trained_model.pth')
numactions = len(env.unwrapped.get_action_meanings())

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cpu")


steps_done = 0
num_episodes = 300
episode_durations = []
prepscreen = T.Compose([#T.ToPILImage(),
                    #T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.maxp = nn.MaxPool2d(4)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        # self.bn4 = nn.BatchNorm2d(32)
        self.predp = nn.Linear(32*5*3, 32*5*3) # this predicts next representation from prev representation
        self.preda = nn.Linear(numactions, 32*5*3) # this predicts next representation from prev action
        self.policy = nn.Linear(32*5*3, numactions) # this generates action

    def forward(self, x, a): # inputs are: x = frame, a = action
        x = self.maxp(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        representation = x # output representation of network
        # print(x.size())
        policy = self.policy(x.view(x.size(0), -1))
        pred = self.predp(x.view(x.size(0), -1)) + self.preda(a)
        return policy, pred, representation


def select_action(state, threshold):
    return policy.max(1)[1].view(1, 1), representation


def get_screen():
    screen = env.render(mode='rgb_array') # size: (3, 210, 160)
    return prepscreen(screen).unsqueeze(0).to(device)


# main script:
target_net = CNN().to(device)
target_net.load_state_dict(torch.load(model_file_name))
target_net.eval()

for i_episode in range(num_episodes):
    observation = env.reset()
    for t in count():
        env.render()
        # print(observation)
        action = env.action_space.sample() # random action
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode', i_episode+1, 'finished after {} timesteps'.format(t+1))
            break

