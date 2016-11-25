-- Eugenio Culurciello
-- November 2016
-- Deep RNN for reinforcement online learning

if not dqn then
    require "initenv"
end
require 'image'
local of = require 'opt' -- options file
local opt = of.parse(arg)
print('Options are:', opt)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)

-- setup game environment:
local gameEnv, gameActions = gameEnvSetup(opt) -- setup game environment
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
    local screen, reward, isGameOver = gameEnv:nextRandomGame()
    screen, reward, isGameOver = gameEnv:step(gameActions[2], true) -- start game!
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
        screen, reward, isGameOver = gameEnv:step(gameActions[action]) -- test mode 
        local nextState = preProcess(screen)

        -- Update the current state and if the game is over:
        currentState = nextState

        win = image.display({image=screen, zoom=2, win=win})
    end

    if isGameOver then 
      episodes = episodes + 1
      print('Episodes: ', episodes, 'total reward:', totalReward)
    end
    collectgarbage()
end
