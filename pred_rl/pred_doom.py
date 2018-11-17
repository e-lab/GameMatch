# E. Culurciello, November 2018
# Predictive neural networks for RL: vizDoom
#
# This uses an Intrinsic Reward = prediction of future representations
# as well as an Extrinsic Reward = A2C, actor-critic reward from game
#
# Algorithm:
# Step 1: frame f_t --CNN1--> embedding e_t --policy--> action a_t
# Step 2: e_t, a_t --pred_net--> e^_t+1
# Step 3: step play: a_t --game--> f_t+1 --CNN1--> e_t+1
# Step 4: minimize ||e_t+1 - e^_t+1||
#
# inspired by: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py

import argparse
import gym
import numpy as np
import itertools as it
from itertools import count
from collections import namedtuple
from termcolor import colored

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# from torchvision.transforms import ToTensor
# from torchvision.transforms import Resize
import skimage.color, skimage.transform

from vizdoom import *

parser = argparse.ArgumentParser(description='PyTorch VizDoom prediction nn example')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


game_name = 'ViZDoom' # Breakout, Pong, Enduro
# env = gym.make(game_name)
print(colored('\nPlaying game:', 'green'), game_name)
# print(colored('available actions:', 'red'), env.unwrapped.get_action_meanings(), '\n')
# numactions = 2 #len(env.unwrapped.get_action_meanings())

# env.seed(args.seed)
torch.manual_seed(args.seed)

eps = np.finfo(np.float32).eps.item() # a small number to avoid division by 0

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10

model_savefile = "./model-doom.pth"
save_model = True
load_model = False
skip_learning = False

# Configuration file path
config_file_path = "/usr/local/lib/python3.6/site-packages/vizdoom/scenarios/simpler_basic.cfg"
# config_file_path = "../../scenarios/rocket_basic.cfg"
# config_file_path = "../../scenarios/basic.cfg"


class CNN1(nn.Module):
# predictive network
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(192, 128)
        # self.fc2 = nn.Linear(128, available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        r = F.relu(self.fc1(x))
        return r

# class Net(nn.Module):
#     def __init__(self, available_actions_count):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
#         self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
#         self.fc1 = nn.Linear(192, 128)
#         self.fc2 = nn.Linear(128, available_actions_count)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(-1, 192)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)


class Pred_NN(nn.Module):
# predictive network
    def __init__(self, numactions):
        super(Pred_NN, self).__init__()
        self.decoder_r = nn.Linear(128, 128)
        self.decoder_action = nn.Linear(numactions, 128)

    def forward(self, x, a):
        p = F.relu(self.decoder_r(x)) + F.relu(self.decoder_action(a)) # prediction of next state
        return p


class Policy_NN(nn.Module):
# actor critic heads
    def __init__(self, numactions):
        super(Policy_NN, self).__init__()
        self.action_head = nn.Linear(128, numactions)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        action_scores = self.action_head(x)
        state_values = self.value_head(x).squeeze()
        return F.softmax(action_scores, dim=-1), state_values




def one_hot_convert(x): # convert action vector to 1-hot vector
    converted = torch.zeros([x.size(0), numactions], dtype=torch.float)
    converted[np.arange(args.batch_size), x] = 1.
    # converted = torch.zeros(numactions)
    # converted[x] = 1
    return converted


def select_action(state):
    r = CNN_model(state)
    probs, state_value = policy_model(r)
    m = Categorical(probs)
    action = m.sample()
    policy_model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    # Step 2: e_t, a_t --pred_net--> e^_t+1
    actt = one_hot_convert(action) # convert to 1-hot
    p = pred_model(r, actt)
    return action.item(), p


def finish_episode():
    R = 0
    saved_actions = policy_model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in policy_model.rewards[::-1]:
        R = r + args.gamma * R # discount all sums of reward
        rewards.insert(0, R) # insert at 0 position (1st position)
    rewards = torch.tensor(rewards)
    # print('rewards: ', rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    # print('normalized rewards: ', rewards)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    # print('policy_losses: ', policy_losses)
    # print('value_losses: ', value_losses)
    optimizer_policy.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # print('loss', loss)
    loss.backward()
    optimizer_policy.step()
    del policy_model.rewards[:]
    del policy_model.saved_actions[:]


def learn_pred(p, s):
    optimizer_pred.zero_grad()
    loss = loss_pred(p, s) 
    optimizer_pred.step()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = img.reshape([1, 1, resolution[0], resolution[1]])
    img = torch.from_numpy(img).float()
    return img


# Create Doom instance
game = initialize_vizdoom(config_file_path)

# Action = which buttons are pressed
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]
numactions = len(actions)

# Models:
# CNN generates representation / encoding:
CNN_model = CNN1()
# Intrinsic Reward - prediction:
pred_model = Pred_NN(numactions)
optimizer_pred = optim.Adam(pred_model.parameters())
loss_pred = nn.MSELoss()
# Extrinsic Reward - A2C / actor-critic:
policy_model = Policy_NN(numactions)
optimizer_policy = optim.Adam(policy_model.parameters())


def main():
    running_reward = 10
    for i_episode in count(1):
        game.new_episode()
        state = preprocess(game.get_state().screen_buffer)
        for t in range(10000):  # Don't infinite loop while learning
            # Step 1: frame f_t --CNN1--> embedding e_t --policy--> action a_t
            action, pred = select_action(state)
            # Step 2 is inside previous function
            # Step 3: step play: a_t --game--> f_t+1 --CNN1--> e_t+1
            # state, reward, done, _ = env.step(action)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()
            # print(reward)
            if args.render:
                env.render()
            policy_model.rewards.append(reward)
            # Step 4: minimize ||e_t+1 - e^_t+1||
            learn_pred(pred, CNN_model(state))  # Intrinsic Reward used - prediction
            if done:
                break
            else:
                state = preprocess(game.get_state().screen_buffer)

        # print(model.rewards)
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode() # Extrinsic Reward used
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        # if running_reward > env.spec.reward_threshold:
            # print("Solved! Running reward is now {} and "
                  # "the last episode runs to {} time steps!".format(running_reward, t))
            # break

    # game.close()

if __name__ == '__main__':
    main()
