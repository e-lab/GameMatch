# Catch game:

This repository is designed to experiment with recent Reinforcement Learning algorithms and implement them in a easy and accessible way. It is designed to be both readable and efficient.



### Q-learning with MLP / CNN:

Train: `th train.lua` trains an MLP model to achieve 100% accuracy. `th train.lua --modelType cnn` trains a CNN model to 100% accuracy.

Test trained network: `th train.lua --skipLearning --load results/model-catch-dqn.net` to load a pre-trained network and run test game. Run with `qlua train.lua --skipLearning --load results/model-catch-dqn.net --display` to see output display. 

To run on a larger game grid (default grid size is 10 x 10) you can use: `th train.lua --gridSize 20 --epochs 20 --learningStepsEpoch 2000` for a 20 x 20 grid size.

Separate test on game play: `qlua test-catch.lua catch-model-grid.net 10`, params: trained CNN/MLP model, grid size (must be same as training), rnn (optional to test RNN models, see below also)


### RNN version:

Uses an RNN to learn successful sequences of moves:

Train: `th train-rnn.lua` (any batch size, and GPU), trains to ~ 100%. This uses a RNN with 2 weight matrices.

Test on game play: `qlua test.lua catch-model-rnn.net 10 rnn`, params: trained RNN model, grid size (must be same as training), rnn (test RNN, otherwise CNN/MLP)


Note:Tested with batches of 128 on GPU and grid size of 25 successful in 1e4 epochs.

Note2: `th train-rnn.lua --gridSize 30 --useGPU --batchSize 128 --epochs 5e4` reported: `Game: 50000, epsilon: 0.02, error: 1.5099, Random Actions: 0, Accuracy: 69%, time [ms]: 11`


RNN + Fast Weight version: a new version is available: `th train-rnn.lua --fw --epoch 1e4`, which delivers ~100% accuracy in 1e4 epochs. Compared to standard RNN only version `th train-rnn.lua --epoch 1e4`, which obtains 94% max in same 1e4 epochs.