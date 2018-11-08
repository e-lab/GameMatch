# E. Culurciello
# November 2018
# experiments with predictive networks in RL
#
# inspired by: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

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

game = 'Breakout-v0'
env = gym.make(game)
print(colored('\nPlaying game:', 'green'), game)
print(colored('available actions:', 'red'), env.unwrapped.get_action_meanings(), '\n')
numactions = len(env.unwrapped.get_action_meanings())

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.set_printoptions(precision=3)
# set up a tensor 1-hot for the action 0 = NOOP:
no_op_action = torch.zeros(numactions, device=device, dtype=torch.float)
no_op_action[0] = 1.


BATCH_SIZE = 256
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 10
MEM_SIZE = 10000
steps_done = 0
num_episodes = 200
episode_durations = []
saved_model_filename = 'saved_model.pth'

prepscreen = T.Compose([T.ToTensor()])
Transition = namedtuple('Transition',
        ('state', 'action', 'representation', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


def one_hot_convert(x): # convert action vector to 1-hot vector
    converted = torch.zeros([x.size(0), numactions], dtype=torch.float)
    converted[np.arange(BATCH_SIZE), x.squeeze(1)] = 1.
    return converted


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    action_batch_tensors = one_hot_convert(action_batch).to(device)
    representation_batch = torch.cat(batch.representation)
    # reward_batch = torch.cat(batch.reward)

    # compute predictions and prediction error (loss)
    policies, preds, _ = policy_net(state_batch, action_batch_tensors)
    representations = representation_batch.reshape(BATCH_SIZE, -1)
    outputf = loss(representations, preds)

    # Optimize the model
    optimizer.zero_grad()
    outputf.backward()
    # for param in policy_net.parameters():
        # param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return outputf


def select_action(state, threshold):
    sample = random.random()
    with torch.no_grad():
            policy, pred,representation  = policy_net(state, no_op_action)
    if sample > threshold:
            return policy.max(1)[1].view(1, 1), representation
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long), representation


def get_screen():
    screen = env.render(mode='rgb_array') # size: (3, 210, 160)
    return prepscreen(screen).unsqueeze(0).to(device)


# main script:
memory = ReplayMemory(MEM_SIZE)
policy_net = CNN().to(device)
target_net = CNN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

loss = nn.MSELoss()
optimizer = optim.RMSprop(policy_net.parameters())

steps_done = 0

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    # img1 = current_screen.squeeze(0).transpose(0,2).numpy()
    # print(img1.shape)
    # plt.imshow(img1)
    # plt.show()
    state = get_screen()
    losst = 0
    for t in count():
        # env.render()
        # update esploration threshold
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        # select and perform an action
        action, representation = select_action(state, eps_threshold)
        # print(action.item())
        _, reward, done, _ = env.step(action.item())
        # reward = torch.tensor([reward], device=device)

        # observe new state
        if not done:
            next_state = get_screen()
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, representation, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        lossi = optimize_model()
        if lossi is not None:
            losst = losst + lossi.item()
        if done:
            episode_duration = t + 1
            print('Steps: {:d}, eps_threshold: {:.2f}, loss: {:.2f}, Episode duration: {:d}'
                .format(steps_done, eps_threshold, losst, episode_duration))
            losst = 0
            break

    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

# final notes:
torch.save(target_net.state_dict(), saved_model_filename) # save trained network
env.close()
print('Training complete, trained network saved as:', saved_model_filename)




