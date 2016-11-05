# Atari

NOT WORKING - DO NOT USE!!!!

## installation

requires [xitari](https://github.com/deepmind/xitari.git) and [alewrap](https://github.com/deepmind/alewrap.git
)

## training a new RL model

`qlua train.lua --display` to display output game locally on CPU.

`qlua train.lua --display --useGPU` to display output game locally on GPU.

`th train.lua --useGPU` to train on remote server with no display on GPU.

Inspired by: https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner

Note: even after a long long time it does not train on breakout!
##To save playinf frames from atari

run play.lua with qlua then it will save frames and action under save folder
