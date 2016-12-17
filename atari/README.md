# Atari


## installation

requires [xitari](https://github.com/deepmind/xitari.git) and [alewrap](https://github.com/deepmind/alewrap.git
)

To install dependencies run bellow with elab account

```
sh installDependencies.sh
```

## training a new DQN / CNN model

`th train.lua`, to train on CPU for 100 epochs.

`th train.lua --useGPU --batchSize 256` to train on GPU (increase batch size here for faster learning).


## training an RNN model

`th train-rnn.lua`



##To save playing frames from atari

run play.lua with qlua then it will save frames and action under save folder
