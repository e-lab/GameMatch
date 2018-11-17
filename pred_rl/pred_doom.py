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
from time import time, sleep
from tqdm import trange

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
epochs = 20
learning_steps_per_epoch = 2000
test_episodes_per_epoch = 100
replay_memory_size = 10000

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

Transition = namedtuple('Transition',
        ('state', 'action', 'log_prob', 'value', 'next_state', 'reward'))

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


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
        state_values = self.value_head(x)
        return F.softmax(action_scores), state_values.squeeze(1)


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

# Create replay memory which will store the transitions
memory = ReplayMemory(capacity=replay_memory_size)

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


def perform_learning_step(epoch, state):
    # Step 1: frame f_t --CNN1--> embedding e_t --policy--> action a_t
    old_state = state
    state = preprocess(game.get_state().screen_buffer)
    action, pred = select_action(state)
    # Step 2 is inside previous function
    # Step 3: step play: a_t --game--> f_t+1 --CNN1--> e_t+1
    reward = game.make_action(actions[action], frame_repeat)
    policy_model.rewards.append(reward)
    done = game.is_episode_finished()
    # add to replay memory:
    memory.add_transition(old_state, action, state, done, reward)
    # Step 4: minimize ||e_t+1 - e^_t+1||
    learn_pred(pred, CNN_model(state))  # Intrinsic Reward used - prediction
    return state


def main():
    print("Starting the training!")
    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            state = preprocess(game.get_state().screen_buffer)
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                state = perform_learning_step(epoch, state)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1
                else:
                    # record non terminal transition in replay memory
                    memory.push(r_t, action, ps[0], ps[1], r_tp1, reward)

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    action,_ = select_action(state)

                    game.make_action(actions[action], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f +/- %.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            # print("Saving the network weigths to:", model_savefile)
            # torch.save(model, model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            action,_ = select_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[action])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)


if __name__ == '__main__':
    main()
