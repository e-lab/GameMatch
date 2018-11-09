# E. Culurciello
# November 2018
# experiments with predictive networks in RL
#
# inspired by: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#
# DEMO code from a pre-trained network

import sys
import gym
import random
from itertools import count
from termcolor import colored
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# parse arguments:
parser = argparse.ArgumentParser(description="Training RL with predictive networks - demo")
parser.add_argument('--eps_end', type=float, default=0.05, help='')
parser.add_argument('--num_episodes', type=int, default=50, help='number games to play')
parser.add_argument('--model', type=str, default='saved_model.pth', help='trained model to load')
args = parser.parse_args()

game = 'Breakout-v0'
env = gym.make(game)
print(colored('\nPlaying game:', 'green'), game)
print('Demo using pre-trained network:', args.model)
print('Usage: python3 demo_trained.py trained_model.pth')
numactions = len(env.unwrapped.get_action_meanings())

# if gpu is to be used
device = torch.device("cpu")

prepscreen = T.Compose([T.ToTensor()])

# set up a tensor 1-hot for the action 0 = NOOP:
no_op_action = torch.zeros(numactions, device=device, dtype=torch.float)
no_op_action[0] = 1.


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
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
    sample = random.random()
    with torch.no_grad():
            policy, pred,representation  = trained_net(state, no_op_action)
    if sample > threshold:
            return policy.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


def get_screen():
    screen = env.render(mode='rgb_array') # size: (3, 210, 160)
    return prepscreen(screen).unsqueeze(0).to(device)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


# main script:
trained_net = CNN().to(device)
trained_net.load_state_dict(torch.load(args.model))
trained_net.eval()

for i_episode in range(args.num_episodes):
    env.reset()
    # _,_,done,_ = env.step(1)
    # if done:
    #     env.reset()
    # _,_,done,_ = env.step(2)
    # if done:
    #     env.reset()
    for t in count():
        env.render()
        state = get_screen()
        action = select_action(state, args.eps_end)
        # print(t,action)
        _, reward, done, info = env.step(action)
        if done:
            print('Episode', i_episode+1, 'finished after {} timesteps'.format(t+1))
            break
        if t > 1000:
            break 

env.close()
print('Finished presenting demo')

