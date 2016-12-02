local o = {}
function o.parse(arg)
   require 'pl'
   local lapp = require 'pl.lapp'
   local opt = lapp [[

     Game options:
     --gridSize            (default 8)          game grid size
     --ch                  (default 1)          game grid size
     --discount            (default 0.9)         discount factor in learning
     --epsilon             (default 1)           initial value of ϵ-greedy action selection
     --epsilonMinimumValue (default 0.02)        final value of ϵ-greedy action selection
     --nbActions           (default 3)           catch number of actions
     --playFile            (default '')          human play file to initialize exp. replay memory

     Training parameters:
     --threads               (default 8)         number of threads used by BLAS routines
     --seed                  (default 1)         initial random seed
     -r,--learningRate       (default 0.1)       learning rate
     -d,--learningRateDecay  (default 1e-9)      learning rate decay
     -w,--weightDecay        (default 0)         L2 penalty on the weights
     -m,--momentum           (default 0.9)       momentum parameter
     --batchSize             (default 64)        batch size for training
     --maxMemory             (default 1e3)       Experience Replay buffer memory
     --epochs                (default 1e4)       number of training steps to perform
     --useGPU                                    use GPU in training
     --gpuId                 (default 1)         which GPU to use
     --nSeq                  (default 10)         Sequenc length

     Model parameters:
     --fw                                        Use FastWeights or not
     --nLayers               (default 1)         RNN layers
     --nHidden               (default 5)       RNN hidden size
     --nFW                   (default 8)         number of fast weights previous vectors

     Display and save parameters:
     --zoom                  (default 4)        zoom window
     -v, --verbose           (default 2)        verbose output
     --display                                  display stuff
     --savedir          (default './results')   subdirectory to save experiments in
     --progFreq              (default 1e2)       frequency of progress output
   ]]
   return opt
end

return o
