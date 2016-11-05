# Catch:


### RNN version:

Uses an RNN to learn successful sequences of moves:

Train: `th catch-rnn.lua --epoch 2e5` (only works for batch=1 for now), trains to > 95%.


### Q-learning with MLP / CNN:

Train: `th train-catch.lua --modelType cnn` is a CNN version and delivers up to 80% accuracy

`th train-catch.lua` runs an MLP version with slightly lower performance

Test: `qlua test-catch.lua catch-model-grid.net 10` to run it in test mode
