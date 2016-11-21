#This is convRNN version

- To test run

```
sh run.sh
```
##Explanation

- ich : input channel
- och : output channel
- seq : number of seq
- nFW : fast weight not supported
- w   : width
- h   : hight
- action : number of action

Embed fully connected layer to map 3D to 1D action

```
gm:getModel(ich, och, nHL, K, seq, nFW, w, h , action)
```
