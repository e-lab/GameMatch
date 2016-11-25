-- Eugenio Culurciello
-- November 2016
-- Deep RNN for reinforcement online learning

if not dqn then
    require "initenv"
end
require 'image'
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
  --pool_frms_type      (default 'max')             pool inputs frames mode
  --pool_frms_size      (default '1')               pool inputs frames size
  --actrep              (default 4)                 how many times to repeat action, frames to skip to speed up game and inference
  --randomStarts        (default 30)                play action 0 between 1 and random_starts number of times at the start of each training episode
 
  Training parameters:
  --threads               (default 8)         number of threads used by BLAS routines
  --seed                  (default 1)         initial random seed
  --useGPU                                    use GPU in training
  --gpuId                 (default 1)         which GPU to use

  Model parameters:
  --loadNet             (default '')                trained neural network to load
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

if opt.verbose >= 1 then
    print('Using options:')
    for k, v in pairs(opt) do
        print(k, v)
    end
end


torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)

local gameEnv, gameActions, agent, opt = setup(opt) -- setup game environment
print('Game started. Number of game actions:', #gameActions)
local nbActions = #gameActions
local nbStates = opt.gridSize * opt.gridSize
local nSeq = 4*opt.gridSize -- RNN max sequence length in this game is grid size


function preProcess(im)
  local net = nn.SpatialMaxPooling(4,4,4,4)
  -- local pooled = net:forward(im[1])
  local pooled = net:forward(im[1][{{},{94,194},{9,152}}])
  -- print(pooled:size())
  local out = image.scale(pooled, opt.gridSize, opt.gridSize):sum(1):div(3)
  return out
end

-- Create the base RNN model:
local RNNh0Proto = {} -- initial state - prototype
local RNNhProto = {} -- state to loop through prototype in inference

require 'cunn'
local prototype = torch.load(opt.loadNet)
prototype = prototype:float() -- convert to CPU for demo

-- Default RNN intial state set to zero:
for l = 1, opt.nLayers do
   RNNh0Proto[l] = torch.zeros(1, opt.nHidden) -- prototype forward does not work on batches
end

-- test model:
print('Testing model and prototype RNN:')
local ttest 
ttest = {torch.Tensor(1, nbStates), torch.Tensor(1, opt.nHidden)} 
-- print(ttest)
local a = prototype:forward(ttest)
-- print('TEST prototype:', a)

local episodes, totalReward = 0, 0

print('Begin playing...') -- and play:
while true do
    sys.tic()
    
    -- Initialise the environment.
    local screen, reward, gameOver = gameEnv:nextRandomGame()
    local currentState = preProcess(screen) -- resize to smaller size
    local isGameOver = false
    -- reset RNN to intial state:
    RNNhProto = table.unpack(RNNh0Proto)
    while not isGameOver do
        local q = prototype:forward({currentState:view(1, nbStates), RNNhProto}) -- Forward the current state through the network.
        RNNhProto = q[1]
        
        -- find best action:
        local max, index = torch.max(q[2][1], 1) -- [2] is the output, [1] is state...
        local action = index[1]
        print(action)
        screen, reward, gameOver = gameEnv:step(gameActions[action]) -- test mode 
        local nextState = preProcess(screen)

        -- Update the current state and if the game is over:
        currentState = nextState
        isGameOver = gameOver

        win = image.display({image=screen, zoom=2, win=win})
    end

    if isGameOver then 
      episodes = episodes + 1
      print('Episodes: ', episodes, 'total reward:', totalReward)
    end
    collectgarbage()
end
