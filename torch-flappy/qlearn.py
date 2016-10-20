#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

# use lua / Torch7 as our Tool:
import lutorpy as lua
require("nn")
torch.setnumthreads(8)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 32. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def buildmodel():
    model = nn.Sequential()
    model._add(nn.SpatialConvolution(img_channels,32,8,8,4,4))
    model._add(nn.ReLU())
    model._add(nn.SpatialConvolution(32,64,4,4,2,2))
    model._add(nn.ReLU())
    model._add(nn.SpatialConvolution(64,64,3,3,1,1))
    model._add(nn.ReLU())
    model._add(nn.View(64*6*6))
    model._add(nn.Linear(64*6*6,512))
    model._add(nn.ReLU())
    model._add(nn.Linear(512,2))

    model.criterion = nn.MSECriterion()

    # print(model._forward(torch.Tensor(4,80,80))) # test
    return model

def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    if args['mode'] == 'Run': # evaluation mode:
        OBSERVE = 999999999 
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        # model.load_weights("model.h5")
        # adam = Adam(lr=1e-6)
        # model.compile(loss='mse',optimizer=adam)
        # print ("Weight load successfully")    
    else: # training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randint(0, ACTIONS-1)
                a_t[action_index] = 1
            else:
                q = model._forward(torch.fromNumpyArray(s_t)._float()) #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q.asNumpyArray())
                action_index = max_Q
                a_t[max_Q] = 1

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2
           
            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = model._forward( torch.fromNumpyArray(state_t)._float() ).asNumpyArray()
                Q_sa = model._forward( torch.fromNumpyArray(state_t1)._float() ).asNumpyArray()

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
            
            # now train model on batch:
            ii = torch.fromNumpyArray(inputs)._float()
            oo = model._forward(ii)
            tt = torch.fromNumpyArray(targets)._float()

            model._zeroGradParameters()
            loss = model.criterion._forward(oo, tt)
            dE_dy = model.criterion._backward(oo, tt)
            model._backward(ii, dE_dy)
            model._updateParameters(0.001)
            # loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        t = t + 1

        # save progress every few iterations
        if t % 10000 == 0:
            print("Saving model")
            torch.save("model_"+str(t)+".net", model._clone()._clearState()._float())

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")

def playGame(args):
    print('Building model...')
    model = buildmodel()
    print('the model is:', model)
    print('Training...')
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    main()
