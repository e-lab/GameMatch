# Reinforcement Learning for video games

## training a new RL model

`qlua train.lua --display` to display output game locally on CPU.

`qlua train.lua --display --useGPU` to display output game locally on GPU.

`th train.lua --useGPU` to train on remote server with no display on GPU. 

Inspired by: https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner


## saving data

use: `play.lua` to save data.

It save frame based on freq, seq, size.

t[#t] = Float Tensor with size x seq x 3 x 210 x 160

It will save if reward is given regardless of sampling freq
