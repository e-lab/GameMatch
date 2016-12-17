# Reinforcement Learning for video games

Our experiments with Q-learning, DQN, RNN and more

Directories:

- atari: DQN for atari video games
- catch: simple catch game for beginners in RL
- torch-flappy: flappy bird in python with Torch7 hooks
- doom: Doom simulator from: https://github.com/Marqt/ViZDoom 




## Status:

### Atari:

- DQN CNN script working
- RNN script some good results, needs more testing


### Catch:

- DQN CNN script working
- RNN script working

### Flappy:

- DQN CNN script functional, all code tested, slow, never fully tested, training takes too long

### Doom:

- DQN CNN script working
- RNN script functional, all code tested, but no good results


## Notes:

Note1: RNN in Catch works on grid of 10 x 10 because we know exactly how long the sequence is going to be it is always gridSize-2! in other games this is not possible

FIXED: just needed to shift sequences so reward is at the end!