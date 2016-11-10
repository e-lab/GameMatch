# Catch:


### RNN version:

Uses an RNN to learn successful sequences of moves:

Train: `th catch-rnn.lua --epoch 2e5` (only works for batch=1 for now), trains to ~ 100%. This uses a RNN with 2 weight matrices.

Test on game play: `qlua test.lua catch-model-rnn.net 10 rnn`, params: trained RNN model, grid size (must be same as training), rnn (test RNN, otherwise CNN/MLP)


### Q-learning with MLP / CNN:

Train: `th train-catch.lua --modelType cnn` is a CNN version and delivers up to 80% accuracy

`th train-catch.lua` runs an MLP version with slightly lower performance

Test on game play: `qlua test-catch.lua catch-model-grid.net 10`, params: trained CNN/MLP model, grid size (must be same as training)
