# Catch:

This repository is designed to experiment with recent Reinforcement Learning algorithms and implement them in a easy and accessible way. It is designed to be readable, and not efficient.

Yet there RNN + fast weights implementation below is much better than Q-Learning with MLP and CNN.


### RNN version:

Uses an RNN to learn successful sequences of moves:

Train: `th catch-rnn.lua --epoch 2e5` (only works for batch=1 for now), trains to ~ 100%. This uses a RNN with 2 weight matrices.

Test on game play: `qlua test.lua catch-model-rnn.net 10 rnn`, params: trained RNN model, grid size (must be same as training), rnn (test RNN, otherwise CNN/MLP)


RNN + Fast Weight version: a new version is available: `th train-rnn.lua --fw --epoch 1e4`, which delivers ~100% accuracy in 1e4 epochs. Compared to standard RNN only version `th train-rnn.lua --epoch 1e4`, which obtains 94% max in same 1e4 epochs.


### Q-learning with MLP / CNN:

Train: `th train.lua --modelType cnn` is a CNN version and delivers up to 80% accuracy

`th train-catch.lua` runs an MLP version with slightly lower performance

Test on game play: `qlua test-catch.lua catch-model-grid.net 10`, params: trained CNN/MLP model, grid size (must be same as training)
