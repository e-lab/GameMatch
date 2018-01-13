#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
import random
from random import sample, randint
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable as V
from tqdm import trange
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import Scale
import cv2
from cortexnet import Cortexnet
from cortexnet_basic_rl import Cortexnet_basic_rl
from cortexnet_a2c import Cortexnet_a2c
import shutil
import math
from torchvision.utils import save_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark=True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Q-learning settings
learning_rate = 0.00001
discount_factor = 0.99
epochs = 200

# Training regime
episodes_per_epoch = 1000

# Other parameters
seq_len=10
frame_repeat = 10
resolution = (120, 160)
episodes_to_watch = 10

model_dir='./model_rl_depth'
if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir)
model_loadfile = "./model_skip1/model_45.pt"
model_savefile = os.path.join(model_dir,"model.pt")
save_model = True
load_model = False
skip_learning = False

# Configuration file path
#config_file_path = "../ViZDoom/scenarios/simpler_basic.cfg"
config_file_path = "../ViZDoom/scenarios/rocket_basic.cfg"
#config_file_path = "../ViZDoom/scenarios/basic.cfg"
#config_file_path = "../ViZDoom/scenarios/deadly_corridor.cfg"

# Converts and down-samples the input image
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = Scale(120) (img)
    img = ToTensor() (img)
    img = img.view(1,3,120,160)
    img = img.cuda()
    return img

def preprocess(state):
    img=state.screen_buffer
    depth=state.depth_buffer
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = Scale(120) (img)
    img = ToTensor() (img)
    img = img.view(1,3,120,160)
    depth = Image.fromarray(depth)
    depth = Scale(120)(depth)
    depth = ToTensor()(depth)
    depth = depth.view(1, 1, 120, 160)
    img=torch.cat((img,depth),1)
    img = img.cuda()
    return img


#criterion = nn.SmoothL1Loss()
criterion=nn.MSELoss()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_depth_buffer_enabled(True)
    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':

    of = open(os.path.join(model_dir,'test.txt'), 'w')

    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    #some_action = [0]*n; some_action[0]=1
    #actions=[list(b) for b in list(set([a for a in it.permutations(some_action)]))]

    if load_model:
        print("Loading model from: ", model_loadfile)
        model = Cortexnet_a2c((120, 160),2)
        my_sd=torch.load(model_loadfile)
        model.load_state_dict(my_sd)
        model.fc1=nn.Linear(256*8*10,1024)
        model.dropout=nn.Dropout(p=0.5)
        model.fc2=nn.Linear(1024,len(actions))
    else:
        model = Cortexnet_a2c((120,160),len(actions))
        model.value = nn.Linear(1024, 1)

    model=model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting the training!")
    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []
            loss_value_total=0.0
            loss_policy_total=0.0
            loss_entropy_total=0.0
            loss_total=0.0
            steps_total=0

            print("Training...")
            model=model.train()
            for learning_step in trange(episodes_per_epoch, leave=False):
                t=0; loss=0.0; loss_value=0.0; loss_policy=0.0
                state=None
                game.new_episode()
                while not game.is_episode_finished():
                    reward_list=[]
                    probs_list=[]
                    log_probs_list=[]
                    value_list=[]
                    loss=0.0
                    for t in range(seq_len):
                        #s1 = preprocess(game.get_state().screen_buffer)
                        s1 = preprocess(game.get_state())
                        (x_hat, policy, value, state) = model(V(s1), state)
                        probs=F.softmax(policy)
                        log_probs=F.log_softmax(policy)
                        m, index = torch.max(probs, 1)
                        a = index.data[0]
                        probs_list.append(probs[0,a])
                        log_probs_list.append(log_probs[0,a])
                        reward = game.make_action(actions[a], frame_repeat)/100.0
                        #print(reward)
                        reward_list.append(reward)
                        value_list.append(value[0,0])
                        isterminal = game.is_episode_finished()
                        if isterminal:
                            break
                    if isterminal:
                        R=0.0
                    else:
                        #s2 = preprocess(game.get_state().screen_buffer)
                        s2 = preprocess(game.get_state())
                        (_, _, v, _) = model(V(s2), state)
                        R = v.data[0, 0]
                    for i in reversed(range(len(reward_list))):
                        R=reward_list[i]+discount_factor*R
                        advantage=R-value_list[i].data[0]
                        loss_policy=-log_probs_list[i]*advantage
                        loss_value=criterion(value_list[i],V(torch.cuda.FloatTensor([R])))
                        #print(value_list[i].data[0])
                        loss_entropy = (-1) * (-1) * (probs_list[i] * log_probs_list[i]).sum()
                        loss += loss_policy+loss_value+0.01*loss_entropy
                        loss_policy_total += loss_policy.data[0]
                        loss_value_total += loss_value.data[0]
                        loss_entropy_total += loss_entropy.data[0]
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), 50.0)
                    optimizer.step()
                    for j in range(len(state)):
                        state[j] = state[j].detach()
                    steps_total += len(reward_list)
                score = game.get_total_reward()
                train_scores.append(score)

            train_scores = np.array(train_scores)
            print("Results: mean: %.2f std: %.2f," % (train_scores.mean(), train_scores.std()), "min: %.2f," % train_scores.min(), "max: %.2f," % train_scores.max())
            print('Loss_policy: %f, loss_value: %f, loss_entropy: %f' % (loss_policy_total/steps_total, loss_value_total/steps_total, loss_entropy_total/steps_total))

            print("\nTesting...")
            test_episode = []
            test_scores = []
            model=model.eval()
            for test_episode in trange(episodes_per_epoch, leave=False):
                state=None
                game.new_episode()
                while not game.is_episode_finished():
                    #s1 = preprocess(game.get_state().screen_buffer)
                    s1 = preprocess(game.get_state())
                    (_, actual_q, _, state) = model(V(s1), state)
                    m, index = torch.max(actual_q, 1)
                    a = index.data[0]
                    game.make_action(actions[a], frame_repeat)
                score = game.get_total_reward()
                test_scores.append(score)

            test_scores = np.array(test_scores)
            print("Results: mean: %.2f std: %.2f," % (test_scores.mean(), test_scores.std()), "min: %.2f" % test_scores.min(), "max: %.2f" % test_scores.max())

            print("Saving the network weigths to:", model_savefile)
            torch.save(model.state_dict(), model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))
            of.write('%d,%f,%f\n' % (epoch + 1, (time() - time_start) / 60.0, test_scores.mean())); of.flush()

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    '''
    # Reinitialize the game with window visible
    game.set_window_visible(False)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    all_scores=np.zeros((episodes_to_watch,),dtype=np.float32)
    model=model.eval()

    for i in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            #state = state.reshape([1, 1, resolution[0], resolution[1]])
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        all_scores[i]=score
        print("Total score: ", score)

    final_score=all_scores.mean()
    print('Final scores is ', final_score)
    '''

