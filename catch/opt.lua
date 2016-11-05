
local o = {}
function o.parse(arg)
   local lapp = require 'pl.lapp'
   local opt = lapp [[
     Game options:
     --discount            (default 0.9)         discount factor in learning
     --epsilon             (default 1)           initial value of ϵ-greedy action selection
     --epsilonMinimumValue (default 0.001)       final value of ϵ-greedy action selection
     --nbActions           (default 3)           catch number of actions
     --useGpu                                    useGpu
     --gpuId               (default 1)           Setup gpu id
     
     Training parameters:
     --threads               (default 8)         number of threads used by BLAS routines
     --seed                  (default 1)         initial random seed
     -r,--learningRate       (default 1e-2)       learning rate
     -d,--learningRateDecay  (default 0)      learning rate decay
     -w,--weightDecay        (default 0)         L2 penalty on the weights
     -m,--momentum           (default 0.9)       momentum parameter
     --gridSize              (default 10)        state is screen resized to this size 
     --hiddenSize            (default 100)       hidden states in neural net
     --batchSize             (default 50)        batch size for training
     --maxMemory             (default 0.5e3)     Experience Replay buffer memory
     --epoch                 (default 1e3)       number of training steps to perform
     --progFreq              (default 1e2)       frequency of progress output
     --modelType             (default 'mlp')     neural net model type: cnn, mlp

     Display and save parameters:
     --zoom                  (default 4)        zoom window
     -v, --verbose           (default 2)        verbose output
     --display                                  display stuff
     --savedir          (default './results')   subdirectory to save experiments in
   ]]
   return opt
end

return o
