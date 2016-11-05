# Torch7-FlappyBird

A single 200 lines of python code to demostrate DQN with Torch7

Please read the following blog for details

https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

Modifier by Eugenio Culurciello to work on Torch7

![](animation1.gif)

# Installation Dependencies:

* Python 2.7
* Torch7
* lutorpy
* pygame
* scikit-image

# How to Run?

Requires: https://github.com/imodpasteur/lutorpy

Train: `python qlearn.py -m ''`

This is a python to Torch7 example, based on this: https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

This code is very slow and we did not run long enough to see it converge. It may need to be converted to GPU!


If you want to train the network from beginning, delete the model.h5 and run qlearn.py -m "Train"
README.md