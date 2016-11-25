-- Eugenio Culurciello
-- November 2016
-- Deep RNN for reinforcement online learning

-- train from a human play/experience data

-- run with:  --playFile filename
-- qlua train-exp.lua --playFile results/play-memory-5019.t7 --epochs 1e3 --display --batchSize 512

if not dqn then
    require "initenv"
end
require 'image'
local of = require 'opt' -- options file
opt = of.parse(arg)
print('Options are:', opt)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)
torch.save(paths.concat(opt.savedir,'opt.t7'),opt)
local trainer = require('train-functions') -- train functions
package.path = '../catch/?.lua;' .. package.path
local rnn = require 'RNN'

-- use GPU, if desired:
if opt.useGPU then
  require 'cunn'
  cutorch.setDevice(opt.gpuId)
  local firstRandInt = torch.random()
  cutorch.manualSeed(firstRandInt) -- set cuda random seed
  print('Using GPU number', opt.gpuId)
end

-- setup game environment
local gameEnv, gameActions = gameEnvSetup(opt)
print('Game started. Number of game actions:', #gameActions)
local nbActions = #gameActions
local nbStates = opt.gridSize * opt.gridSize
local nSeq = 4*opt.gridSize -- RNN max sequence length in this game is grid size

-- Create the base RNN model:
local model, prototype
local RNNh0Batch = {} -- initial state for training batches
local RNNh0Proto = {} -- initial state - prototype
local RNNhProto = {} -- state to loop through prototype in inference

if opt.fw then
  print('Created fast-weights RNN with:\n- input size:', nbStates, '\n- number hidden:', opt.nHidden, 
    '\n- layers:', opt.nLayers, '\n- output size:', nbActions, '\n- sequence length:', nSeq, 
    '\n- fast weights states:', opt.nFW)
  model, prototype = rnn.getModel(nbStates, opt.nHidden, opt.nLayers, nbActions, nSeq, opt.nFW)
else
  print('Created RNN with:\n- input size:', nbStates, '\n- number hidden:', opt.nHidden, 
      '\n- layers:', opt.nLayers, '\n- output size:', nbActions, '\n- sequence lenght:',  nSeq)
  model, prototype = rnn.getModel(nbStates, opt.nHidden, opt.nLayers, nbActions, nSeq)
end

-- send all networks to GPU:
if opt.useGPU then
  model = model:cuda()
  prototype = prototype:cuda()
  trainer.criterion = trainer.criterion:cuda()
end

-- Default RNN intial state set to zero:
for l = 1, opt.nLayers do
   RNNh0Batch[l] = torch.zeros(opt.batchSize, opt.nHidden)
   RNNh0Proto[l] = torch.zeros(1, opt.nHidden) -- prototype forward does not work on batches
   if opt.useGPU then RNNh0Batch[l]=RNNh0Batch[l]:cuda() RNNh0Proto[l]=RNNh0Proto[l]:cuda() end
end

-- test model:
print('Testing model and prototype RNN:')
local ttest 
if opt.useGPU then ttest = {torch.CudaTensor(1, nbStates), torch.CudaTensor(1, opt.nHidden)}
else ttest = {torch.Tensor(1, nbStates), torch.Tensor(1, opt.nHidden)} end
-- print(ttest)
local a = prototype:forward(ttest)
-- print('TEST prototype:', a)
if opt.useGPU then ttest = {torch.CudaTensor(opt.batchSize, nSeq, nbStates), torch.CudaTensor(opt.batchSize, opt.nHidden)}
else ttest = {torch.Tensor(opt.batchSize, nSeq, nbStates), torch.Tensor(opt.batchSize, opt.nHidden)} end
-- print(ttest)
local a = model:forward(ttest)
-- print('TEST model:', a)


-- setup memory and training variables:
local memory = Memory(opt.maxMemory, opt.discount)
local seqMem = torch.Tensor(nSeq, nbStates) -- store sequence of states in successful run
local seqAct = torch.zeros(nSeq, nbActions) -- store sequence of actions in successful run
local epsilon = opt.epsilon -- this will change in the training, so we copy it
local epsUpdate = (epsilon - opt.epsilonMinimumValue)/opt.epochs
local winCount = 0
local err = 0
local randomActions = 0




print('Begin training / testing:')

for te = 1, 10 do

  -- train network on play game saved file data:
  for game = 1, opt.epochs do
    sys.tic()
    -- We get a batch of training data to train the model:
    local inputs, targets = memory.getBatch(opt.batchSize, nSeq, nbActions, nbStates)
    -- Train the network which returns the error:
    err = err + trainer.trainNetwork(model, RNNh0Batch, inputs, targets, nSeq, nbActions)
    if game%opt.progFreq == 0 then 
        print(string.format("Training sequence: %d, error: %.4f, time [ms]: %d", 
                             game,  err/opt.progFreq, sys.toc()*1000))
        err = 0
    end
    collectgarbage()
  end

  -- then test 100 games:
  for game = 1, 100 do
    sys.tic()

    local screen, reward, isGameOver = gameEnv:nextRandomGame()
    screen, reward, isGameOver = gameEnv:step(gameActions[2], true) -- start game!
    local currentState = trainer.preProcess(screen) -- resize to smaller size
    -- reset RNN to intial state:
    RNNhProto = table.unpack(RNNh0Proto)
    while not isGameOver do
        if opt.useGPU then currentState = currentState:cuda() end
        prototype:forward({currentState:view(1, nbStates), table.unpack(RNNh0Proto)}) -- HAVE TO DO THIS to clear proto state, or crashes
        local q = prototype:forward({currentState:view(1, nbStates), RNNhProto}) -- Forward the current state through the network.
        RNNhProto = q[1]
        -- make move:
        -- Find the max index (the chosen action).
        local max, index = torch.max(q[2][1], 1) -- [2] is the output, [1] is state...
        local action = index[1]
        
        screen, reward, isGameOver = gameEnv:step(gameActions[action], true)
        local nextState = trainer.preProcess(screen)

        currentState = nextState

        if opt.display then 
            -- win = image.display({image=currentState:view(opt.gridSize,opt.gridSize), zoom=10, win=win})
            win2 = image.display({image=screen, zoom=1, win=win2})
        end
    end

    if game%opt.progFreq == 0 then 
        print(string.format("Game: %d, epsilon: %.2f, Accuracy: %d%%, time [ms]: %d", 
                             game,  epsilon, winCount/opt.progFreq*100, sys.toc()*1000))
        winCount = 0
    end
    collectgarbage()
  end
end
torch.save(opt.savedir.."/proto-rnn.net", prototype:float():clearState())
print("Prototype model saved!")