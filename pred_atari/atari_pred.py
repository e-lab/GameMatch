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
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical


# parse arguments:
parser = argparse.ArgumentParser(description="Training RL with predictive networks")
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
parser.add_argument('--eps_start', type=float, default=0.9, help='')
parser.add_argument('--eps_end', type=float, default=0.05, help='')
parser.add_argument('--eps_decay', type=int, default=200, help='')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--mem_size', type=int, default=20000, help='replay memory size')
parser.add_argument('--num_episodes', type=int, default=300, help='number games to play')
args = parser.parse_args()

game_name = 'Breakout-v0' # Breakout, Pong, Enduro
env = gym.make(game_name)
print(colored('\nPlaying game:', 'green'), game_name)
print(colored('available actions:', 'red'), env.unwrapped.get_action_meanings(), '\n')
numactions = len(env.unwrapped.get_action_meanings())

env.seed(args.seed)
torch.manual_seed(args.seed)
np.set_printoptions(precision=3)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up a tensor 1-hot for the action 0 = NOOP:
no_op_action = torch.zeros(numactions, device=device, dtype=torch.float)
no_op_action[0] = 1.

prepscreen = T.Compose([T.ToTensor()])

Transition = namedtuple('Transition',
        ('state', 'action', 'next_state', 'reward'))

# SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
# eps = np.finfo(np.float32).eps.item()


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


def one_hot_convert(x): # convert action vector to 1-hot vector
    converted = torch.zeros([x.size(0), numactions], dtype=torch.float)
    converted[np.arange(args.batch_size), x] = 1.
    return converted

def get_screen():
    screen = env.render(mode='rgb_array') # size: (3, 210, 160)
    return prepscreen(screen).unsqueeze(0).to(device)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


# neural nets:

class random_NN(nn.Module):
# fixed and random target network: frame_t --> r_t
    def __init__(self):
        super(random_NN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # self.policy = nn.Linear(384, numactions) # this generates action

    def forward(self, x): # inputs are: x = frame
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.size())
        return x.view(x.size(0), -1)

class predict_NN(nn.Module):
# predictor network: r_t, a_t--> r_t+1
    def __init__(self):
        super(predict_NN, self).__init__()
        self.preda = nn.Linear(numactions, 2816) # this predicts next representation from prev action
        self.predp = nn.Linear(2816, 2816) # this predicts next representation from prev representation

    def forward(self, x, a): # inputs are: [r_t, a_t]
        r_tp1 = self.predp(x) + self.preda(a)
        return r_tp1

class policy_NN(nn.Module):
# policy network: from rep_t --> a_t, value_t (actor critic)
    def __init__(self):
        super(policy_NN, self).__init__()
        self.action_head = nn.Linear(2816, numactions) # this generates action
        self.value_head = nn.Linear(2816, 1)

    def forward(self, x):
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


# optimizer routines:

def optimize():   
    if len(memory) < args.batch_size:
        return
    
    #prepare batches:
    transitions = memory.sample(args.batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          # batch.next_state)), dtype=torch.uint8)
    # non_final_next_states = torch.cat([s for s in batch.next_state
                                                # if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch_nt = torch.cat(batch.action)
    action_batch = one_hot_convert(action_batch_nt).to(device)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    # compute predictions and prediction error (loss)
    preds = predict_net(state_batch, action_batch)

    # Optimize pred
    predict_net.train()
    optimizer_pred.zero_grad()
    loss = loss_pred(next_state_batch, preds)
    loss.backward()
    for param in predict_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer_pred.step()


    # optimize policy: using actor critic
    # mainimize surprise?

    # R = 0
    # saved_actions = policy_net.saved_actions
    # policy_losses = []
    # value_losses = []
    # rewards = []
    # for r in policy_net.rewards[::-1]:
    #     R = r + args.gamma * R
    #     rewards.insert(0, R)
    # rewards = torch.tensor(rewards)
    # rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    # for (log_prob, value), r in zip(saved_actions, rewards):
    #     reward = r - value.item()
    #     policy_losses.append(-log_prob * reward)
    #     value_losses.append(F.smooth_l1_loss(value, torch.tensor([[r]])))

    # policy_net.train()
    # optimizer_policy.zero_grad()
    # loss_policy = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # loss_policy.backward()
    # optimizer_policy.step()
    # del policy_net.rewards[:]
    # del policy_net.saved_actions[:]


    return loss


def select_action(state):
    # policy net needs to train, target net does not train
    with torch.no_grad():
        r_t = random_net(state)
    probs, state_value = policy_net(r_t)
    m = Categorical(probs)
    action = m.sample()
    # policy_net.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item(), r_t



# main script:
memory = ReplayMemory(args.mem_size)
random_net = random_NN().to(device)
random_net.eval()
predict_net = predict_NN().to(device)
policy_net = policy_NN().to(device)

optimizer_pred = optim.RMSprop(predict_net.parameters())
optimizer_policy = optim.Adam(policy_net.parameters(), lr=3e-2)
loss_pred = nn.MSELoss()



for i_episode in count():
    # Initialize the environment and state
    env.reset()
    l1=0
    l2=0
    running_reward = 0
    state = get_screen()
    for t in range(10000): # Don't infinite loop while learning
        if args.render: 
            env.render()
   
        # select and perform an action
        a_t, r_t = select_action(state)
        _, reward, done, _ = env.step(a_t)
        running_reward = running_reward + reward
        reward = torch.tensor([reward], device=device)
        action = torch.tensor([a_t], device=device)
        
        # observe new state
        if not done:
            next_state = get_screen()
            with torch.no_grad():
                r_tp1 = random_net(next_state)
            # print('memory: ', r_t.size(), a_t, r_tp1.size(), reward)
            memory.push(r_t, action, r_tp1, reward)
            
        # show((state-next_state).squeeze(0))
        # input("Press Enter to continue...")
        
        # Move to the next state
        state = next_state
        if done:
            break

    # optimize predictive network, then at a target loss rate, freexe it!
    loss = optimize()
    if loss is not None:
        l1 = l1+loss.item()
    # if loss_policy is not None:
        # l2 = l2+loss_policy.item()

    # report results:
    if i_episode % args.log_interval == 0:
        print('Episode: {:4d}, loss_pred: {:.2f}, loss_policy: {:.2f}, average length: {:.2f}, average reward: {:.2f}'
            .format(i_episode, l1/args.log_interval, l2/args.log_interval, 
                (t+1)/args.log_interval, running_reward/args.log_interval))
    # if running_reward > env.spec.reward_threshold:
    #     print("Solved! Running reward is now {} and "
    #           "the last episode runs to {} time steps!".format(running_reward, t+1))
    #     break


# final notes:
target_net.to("cpu")
target_net.eval()
saved_model_filename = str(game_name) + '_model.pth'
torch.save(target_net.state_dict(), saved_model_filename) # save trained network
env.close()
print('Training complete, trained network saved as:', saved_model_filename)




