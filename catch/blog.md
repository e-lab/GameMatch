# Thoughts on reinforcement learning and transfer learning to tackle complex tasks

###Eugenio Culurciello, November 2016

## Introduction

It would be nice to be able to teach robots to do things for us: go fetch my keys, make me a sandwich, do the laundry. And we have dextrous robots that can perform [these tasks now](https://en.wikipedia.org/wiki/DARPA_Robotics_Challenge), but what is missing is their brain. And when we talk about brain we are talking about two issues:

- what architecture
- how to train it

We could let our robot explore the environment with little information about it, but it will take a long time to figure things out.

Obviously we would not want to wait for years to train our robot. We want it to learn fast and help us. Ideally we want to train it on YouTube video and by showing examples of what we want. We can call this 'transfer learning' because we need to transfer the knowledge we have to the machine.

When we learn a new task, we do not have to train our brain from a blank slate. Rather we have something that is already pre-trained, and often we follow simple instruction from others on how to perform the task.

Accuracy at the expense of very large or infinite time is not an option. In my opinion it is thus worth to pursue the following goal:

### Goal: Minimize transfer learning of information and trials and maximize accuracy:

-	 accuracy / training steps
-	 accuracy / transferred information

We need an architecture and training that can support transferring information and also obtain good accuracy in the task. Let's see what we can research!

Few of us have access to a decent robot, and it takes time to get physical things to work, so instead we will play with virtual worlds in our computer. Video Games. 

### This sounds exciting! until you realize your computer is now playing the games, and you are doing all the hard work to train it! 

Another reason to be better at the AI game and transfer learning: so we have to do less programming and get more results!


## Q-Learning / Reinforcement Learning

Q-learning and [DeepMind work on DQN](https://arxiv.org/abs/1312.5602) showed how we can train a neural network to play video games by just looking at the screen and score. 

there are many blog post that can help to understand this technique:

- http://mnemstudio.org/path-finding-q-learning-tutorial.htm
- https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html
- https://edersantana.github.io/articles/keras_rl/

Here they used a neural network to learn to play a game with no instruction give (no transfer learning) based on screen state and score only.

Obviously this is a bit of a brute force approach, since the neural network has to learn to do visual tasks AND ALSO play a game at the same time. Building a neural network that can interpret screens should be done ahead of time with pre-training. And helped by transfer learning as much as possible.

Q-Learning is performed online while initially choosing random moves and later replying more and more on a neural network to  offer the best move to get to some reward. We want to get to a Q function that can take in a state (sequence of screen views) and output best next move.

The biggest problem here is that the reward occurs rarely because of random moves, and it can take a long time to find a way to play the game effectively to get rewards, meaning to minimize the number of moves to get to a reward. This can also be mitigated with transfer learning: pre-training with data from human play.

In summary transfer learning can help more here in two forms:

- pre-trained visual neural network
- data from human play

Otherwise Q-learning takes a long time, it is an efficient way to learn to perform a task.


## Catch experiments:

###Q-learning: 

Slow!!! 

Maybe better with transfer learning

### RNN training:

Good initial results!

1- Need transfer learning data.

MatchNet can be trained with #1 to learn visual data unsupervised and also learn sequences to reward





