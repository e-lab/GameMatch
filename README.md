# Reinforcement Learning for video games

Our experiments with Q-learning and DQN and more

Directories:

- atari: DQN for atari video games
- catch: simple catch game for beginners in RL
- torch-flappy: flappy bird in python with Torch7 hooks


# Atari

## installation

requires [xitari](https://github.com/deepmind/xitari.git) and [alewrap](https://github.com/deepmind/alewrap.git
)

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

Train: `th train-catch.lua --modelType cnn` is a CNN version and delivers up to 80% accuracy

`th train-catch.lua` runs an MLP version with slightly lower performance


Test: `qlua test-catch.lua catch-model-grid.net 10` to run it in test mode



# torch-flappy

Requires: https://github.com/imodpasteur/lutorpy

Train: `python qlearn.py -m ''`

This is a python to Torch7 example, based on this: https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

This code is very slow and we did not run long enough to see it converge. It may need to be converted to GPU!


# Notes:

A great library for RL with Torch and openai gym is:
https://github.com/ludc/rltorch

Doom for OS X: https://github.com/soumith/ViZDoom


