# E. Culurciello
# November 2018
# experiments with predictive networks in RL
#
# inspired by: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gym
import math
import random
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

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
steps_done = 0
num_episodes = 50
episode_durations = []
prepscreen = T.Compose([#T.ToPILImage(),
                    #T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
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
        self.pred = nn.Linear(32*5*3, 32*5*3) # this predicts next representation
        self.policy = nn.Linear(32*5*3, 4) # this generates action

    def forward(self, x):
        x = self.maxp(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        representation = x # output representation of network
        # print(x.size())
        policy = self.policy(x.view(x.size(0), -1))
        pred = self.pred(x.view(x.size(0), -1))
        return policy, pred, representation



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    representation_batch = torch.cat(batch.representation)
    reward_batch = torch.cat(batch.reward)


    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    policies, preds, _ = policy_net(state_batch)
    # state_action_values = policies.gather(1, action_batch)

    # # Q-learning not used here - EC!
    # Compute V(s_{t+1}) for all next states.
    # next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # policiest, predt = target_net(non_final_next_states)
    # next_state_values[non_final_mask] = policiest.max(1)[0].detach()
    # Compute the expected Q values
    # expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # predictive network:
    representations = representation_batch.reshape(BATCH_SIZE, -1)
    outputf = loss(representations, preds)

    # Optimize the model
    optimizer.zero_grad()
    outputf.backward()
    # for param in policy_net.parameters():
        # param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return outputf


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    with torch.no_grad():
            policy, pred,representation  = policy_net(state)
    if sample > eps_threshold:
            return policy.max(1)[1].view(1, 1), representation
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long), representation


def get_screen():
    screen = env.render(mode='rgb_array') # size: (3, 210, 160)
    return prepscreen(screen).unsqueeze(0).to(device)


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



# main script:
memory = ReplayMemory(10000)
policy_net = CNN().to(device)
target_net = CNN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

loss = nn.MSELoss()
optimizer = optim.RMSprop(policy_net.parameters())

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    # last_screen = get_screen()
    # current_screen = get_screen()
    # print(current_screen.shape, current_screen.max(), current_screen.min())
    # img1 = current_screen.squeeze(0).transpose(0,2).numpy()
    # print(img1.shape)
    # plt.imshow(img1)
    # plt.show()
    # state = current_screen - last_screen
    state = get_screen()
    losst = 0
    for t in count():
        env.render()
        # Select and perform an action
        # action = env.action_space.sample() # random action
        # print(action)
        # observation, reward, done, info = env.step(action)
        action, representation = select_action(state)
        # print(action.item())
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        # last_screen = current_screen
        # current_screen = get_screen()
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
            episode_durations.append(t + 1)
            # plot_durations()
            print('Loss in this episode:', losst)
            losst = 0
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())




# # example simple plot:
# for i_episode in range(10):
#     observation = env.reset()
#     for t in range(25):
#         env.render()
#         # print(observation)
#         action = env.action_space.sample() # random action
#         observation, reward, done, info = env.step(action)
#         if done:
#             print('Episode', i_episode+1, 'finished after {} timesteps'.format(t+1))
#             break
