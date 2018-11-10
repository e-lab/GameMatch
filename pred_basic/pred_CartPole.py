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
from collections import namedtuple
from termcolor import colored
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# parse arguments:
parser = argparse.ArgumentParser(description="Training RL with predictive networks")
parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
parser.add_argument('--eps_start', type=float, default=0.9, help='')
parser.add_argument('--eps_end', type=float, default=0.05, help='')
parser.add_argument('--eps_decay', type=int, default=200, help='')
parser.add_argument('--target_update', type=int, default=10, help='')
parser.add_argument('--mem_size', type=int, default=20000, help='replay memory size')
parser.add_argument('--num_episodes', type=int, default=300, help='number games to play')
args = parser.parse_args()

game_name = 'CartPole-v0'
env = gym.make(game_name).unwrapped
print(colored('\nPlaying game:', 'green'), game_name)
numactions = 2

np.set_printoptions(precision=3)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up a tensor 1-hot for the action 0 = NOOP:
no_op_action = torch.zeros(numactions, device=device, dtype=torch.float)
no_op_action[0] = 1.


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
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


class target_NN(nn.Module):
# fixed and random target network: from frame to representation and action/policy
    def __init__(self):
        super(target_NN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.policy = nn.Linear(384, numactions) # this generates action

    def forward(self, x): # inputs are: x = frame
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        representation = x.clone() # output representation of network
        # print(x.size())
        probs = self.policy(x.view(x.size(0), -1))
        return probs, representation


class predict_NN(nn.Module):
# predictor network: from frame, action pair --> next representation
    def __init__(self):
        super(predict_NN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.preda = nn.Linear(numactions, 384) # this predicts next representation from prev action
        self.predp = nn.Linear(384, 384) # this predicts next representation from prev representation

    def forward(self, x, a): # inputs are: x = frame, a = action
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.size())
        pred = self.predp(x.view(x.size(0), -1)) + self.preda(a)
        return pred


def one_hot_convert(x): # convert action vector to 1-hot vector
    converted = torch.zeros([x.size(0), numactions], dtype=torch.float)
    converted[np.arange(args.batch_size), x.squeeze(1)] = 1.
    return converted


def optimize_model():
    if len(memory) < args.batch_size:
        return
    transitions = memory.sample(args.batch_size)
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
    preds = predict_net(state_batch, action_batch_tensors)
    representations = representation_batch.reshape(args.batch_size, -1)
    outputf = loss(representations, preds)

    # Optimize the model
    optimizer.zero_grad()
    outputf.backward()
    for param in predict_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return outputf


def select_action(state, threshold):
    sample = random.random()
    with torch.no_grad():
            probs, representation  = target_net(state)
    if sample > threshold:
            return probs.max(1)[1].view(1, 1), representation
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long), representation



# This is based on the code from gym.
screen_width = 600


def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


# main script:
memory = ReplayMemory(args.mem_size)
target_net = target_NN().to(device)
predict_net = predict_NN().to(device)
target_net.eval()

loss = nn.MSELoss()
optimizer = optim.RMSprop(predict_net.parameters())

steps_done = 0

for i_episode in range(args.num_episodes):
    # Initialize the environment and state
    env.reset()
    # img1 = current_screen.squeeze(0).transpose(0,2).numpy()
    # print(img1.shape)
    # plt.imshow(img1)
    # plt.show()
    state = get_screen()
    losst = 0
    for t in count():
        if device == 'cpu': 
            env.render()
        # update esploration threshold
        eps_threshold = 0.05#args.eps_end + (args.eps_start - args.eps_end) * \
            # math.exp(-1. * steps_done / args.eps_decay)
        steps_done += 1
        # select and perform an action
        action, representation = select_action(state, eps_threshold)
        # print(action.item())
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

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
            print('Episode: {:d}, eps_threshold: {:.2f}, loss: {:.2f}, Episode duration: {:d}'
                .format(i_episode, eps_threshold, losst, episode_duration))
            losst = 0
            break


# final notes:
target_net.to("cpu")
target_net.eval()
saved_model_filename = str(game_name) + '_model.pth'
torch.save(target_net.state_dict(), saved_model_filename) # save trained network
env.close()
print('Training complete, trained network saved as:', saved_model_filename)




