# Doom reinforcement learning

Experiments and code by E. Culurciello, Fall 2016


# Code instructions:

## DQN with CNN on Doom:

`th learning-torch7-cnn.lua` to train a neural network from scratch, and test it

`th learning-torch7-cnn.lua --skipLearning` to only test a pre-trained neural network

## RNN on Doom:




# Installation:

## Linux / Ubuntu:

Linux installation issues:

Would not run on remote server with x11, so followed: https://github.com/Marqt/ViZDoom/issues/36

Before `game.init()` added `game.add_game_args("+vid_forcesurface 1")`

solved the problem and runs



## Apple OS X:

`brew install python3`, also use pip for python3 as here:

http://stackoverflow.com/questions/20082935/how-to-install-pip-for-python3-on-mac-os-x


In CMakeLists.txt comment out lines 222-235: https://github.com/Marqt/ViZDoom/blob/master/CMakeLists.txt#L222
to make it compile on OS X 10.12 (Sierra)


Then:


```
brew install boost

brew install boost-python --with-python3 --without-python

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON3=ON -DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python3/3.5.2_3/Frameworks/Python.framework/Versions/3.5/include/python3.5m -DPYTHON_LIBRARY=/usr/local/Cellar/python3/3.5.2_3/Frameworks/Python.framework/Versions/3.5/lib/libpython3.5m.dylib

make

rm bin/vizdoom

ln -s vizdoom.app/Contents/MacOS/vizdoom bin/vizdoom

cd examples/python

python basic.py
```

Look at issues:

https://github.com/Marqt/ViZDoom/issues/147

https://github.com/Marqt/ViZDoom/issues/144

etc.


## Torch7 lua bindings OS X

If you want to build against luajit installed locally by torch (as in http://torch.ch/docs/getting-started.html#_), please do:
```
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=OFF -DBUILD_LUA=ON -DLUA_EXECUTABLE=/Users/eugenioculurciello/torch/bin/luajit -DLUA_LIBRARIES=/Users/eugenioculurciello/torch/install/lib/libluajit.dylib -DLUA_INCLUDE_DIR=/Users/eugenioculurciello/torch/install/include/
```
Then manually copied folder: 

/Users/eugenioculurciello/Desktop/ViZDoom/bin/lua/vizdoom

to:

/Users/eugenioculurciello/torch/install/lib/

and

/Users/eugenioculurciello/torch/install/share/lua/5.1 





