#This is convRNN version

- To test run

```
th testRNNconv.lua
```
#To train convRNN verion

```
th train-convRNN.lua --useGPU
```
##Explanation

- n : input channel
- d : output channel
- nHL : number of layer
- T : number of seq
- nFW : number of fastWeight

Embed fully connected layer to map 3D to 1D action

```
gm.getModel(n, d, nHL, K, T, nFW)
```
