# E. Culurciello, November 2018
# Predictive neural networks for RL:
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
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

eps = np.finfo(np.float32).eps.item() # a small number to avoid division by 0

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class CNN1(nn.Module):
# predictive network
    def __init__(self):
        super(CNN1, self).__init__()
        self.encoder = nn.Linear(4, 128)

    def forward(self, x):
        r = F.relu(self.encoder(x)) # encoding / representation
        return r

class Pred_NN(nn.Module):
# predictive network
    def __init__(self):
        super(Pred_NN, self).__init__()
        self.decoder_r = nn.Linear(128, 4)
        self.decoder_action = nn.Linear(2, 4)

    def forward(self, x, a):
        p = F.relu(self.decoder_r(x)) + F.relu(self.decoder_action(a)) # prediction of next state
        return p


class Policy_NN(nn.Module):
# actor critic heads
    def __init__(self):
        super(Policy_NN, self).__init__()
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


CNN_model = CNN1()
pred_model = Pred_NN()
policy_model = Policy_NN()
optimizer = optim.Adam(policy_model.parameters(), lr=3e-2)
optimizer_pred = optim.Adam(pred_model.parameters(), lr=3e-2)
loss_pred = nn.MSELoss()


def select_action(state):
    state = torch.from_numpy(state).float() # turn state into tensor
    r = CNN_model(state)
    probs, state_value = policy_model(r)
    m = Categorical(probs)
    action = m.sample()
    policy_model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    # Step 2: e_t, a_t --pred_net--> e^_t+1
    p = pred_model(r, probs)
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
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # print('loss', loss)
    loss.backward()
    optimizer.step()
    del policy_model.rewards[:]
    del policy_model.saved_actions[:]


def learn_pred(p, s):
    st = torch.from_numpy(s).float() # turn state into tensor
    optimizer_pred.zero_grad()
    loss = loss_pred(p, st) 
    optimizer_pred.step()


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            # Step 1: frame f_t --CNN1--> embedding e_t --policy--> action a_t
            action, pred = select_action(state)
            # Step 2 is inside previous function
            # Step 3: step play: a_t --game--> f_t+1 --CNN1--> e_t+1
            state, reward, done, _ = env.step(action)
            # print(reward)
            if args.render:
                env.render()
            policy_model.rewards.append(reward)
            # Step 4: minimize ||e_t+1 - e^_t+1||
            learn_pred(pred, state)
            if done:
                break

        # print(model.rewards)
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
