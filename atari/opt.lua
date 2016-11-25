local o = {}
function o.parse(arg)

   require 'pl'
   lapp = require 'pl.lapp'
   opt = lapp [[

     Game options:
     --gridSize            (default 20)          game grid size 
     --discount            (default 0.9)         discount factor in learning
     --epsilon             (default 1)           initial value of ϵ-greedy action selection
     --epsilonMinimumValue (default 0.001)       final value of ϵ-greedy action selection
     --playFile            (default '')          human play file to initialize exp. replay memory
     --framework           (default 'alewrap')         name of training framework
     --env                 (default 'breakout')        name of environment to use')
     --game_path           (default 'roms/')           path to environment file (ROM)
     --env_params          (default 'useRGB=true')     string of environment parameters
     --pool_frms_type      (default "max")             pool inputs frames mode
     --pool_frms_size      (default "1")               pool inputs frames size
     --actrep              (default 4)                 how many times to repeat action, frames to skip to speed up game and inference
     --randomStarts        (default 30)                play action 0 between 1 and random_starts number of times at the start of each training episode
    
     Training parameters:
     --threads               (default 8)         number of threads used by BLAS routines
     --seed                  (default 1)         initial random seed
     -r,--learningRate       (default 0.1)       learning rate
     -d,--learningRateDecay  (default 1e-9)      learning rate decay
     -w,--weightDecay        (default 0)         L2 penalty on the weights
     -m,--momentum           (default 0.9)       momentum parameter
     --batchSize             (default 64)        batch size for training
     --maxMemory             (default 1e3)       Experience Replay buffer memory
     --epochs                (default 1e5)       number of training steps to perform
     --useGPU                                    use GPU in training
     --gpuId                 (default 1)         which GPU to use

     Model parameters:
     --fw                                        Use FastWeights or not
     --nLayers               (default 1)         RNN layers
     --nHidden               (default 128)       RNN hidden size
     --nFW                   (default 8)         number of fast weights previous vectors

     Display and save parameters:
     --zoom                  (default 4)        zoom window
     -v, --verbose           (default 2)        verbose output
     --display                                  display stuff
     --savedir          (default './results')   subdirectory to save experiments in
     --progFreq              (default 1e2)       frequency of progress output
   ]]
   
   -- format options:
   opt.pool_frms = 'type=' .. opt.pool_frms_type .. ',size=' .. opt.pool_frms_size

   return opt
end

return o