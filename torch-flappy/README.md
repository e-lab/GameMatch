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

**CPU only**

```
git clone https://github.com/yanpanlau/Keras-FlappyBird.git
cd Keras-FlappyBird
python qlearn.py -m "Run"
```

**GPU version (Theano)**

```
git clone https://github.com/yanpanlau/Keras-FlappyBird.git
cd Keras-FlappyBird
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.2 python qlearn.py -m "Run"
```

If you want to train the network from beginning, delete the model.h5 and run qlearn.py -m "Train"
