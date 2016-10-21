# Reinforcement Learning for video games

Our experiments with Q-learning and DQN and more

Directories:

- atari: DQN for atari video games
- catch: simple catch game for beginners in RL
- torch-flappy: flappy bird in python with Torch7 hooks


# Atari

## installation

requires xitari and alewrap 

## training a new RL model

`qlua train.lua --display` to display output game locally on CPU.

`qlua train.lua --display --useGPU` to display output game locally on GPU.

`th train.lua --useGPU` to train on remote server with no display on GPU. 

Inspired by: https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner

Note: even after a long long time it does not train on breakout!

## saving data

use: `play.lua` to save data.

It save frame based on freq, seq, size.

t[#t] = Float Tensor with size x seq x 3 x 210 x 160

It will save if reward is given regardless of sampling freq




# Catch:

Requires: https://github.com/Kaixhin/rlenvs

To play game: `qlua play-catch.lua`

Train: `train-catch.lua` same training code as for atari (never managed to get it working)

Train: `train-catch-new.lua`is inspired by: https://github.com/SeanNaren/QlearningExample.torch and blog post: https://edersantana.github.io/articles/keras_rl/


Note: we used convolutional neural nets, as opposed to standard neural nets in our code. The CNN do not train as well and fast, and with small game grids of 10x10 it converges, but with larger grids, such as 24x24 it did not converge / learned to play well after 1e5 iterations~!



# torch-flappy

Requires: https://github.com/imodpasteur/lutorpy

Train: `python qlearn.py -m ''`

This is a python to Torch7 example, based on this: https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

This code is very slow and we did not run long enough to see it converge. It may need to be converted to GPU!


# Notes:

A great library for RL with Torch and openai gym is:
https://github.com/ludc/rltorch
