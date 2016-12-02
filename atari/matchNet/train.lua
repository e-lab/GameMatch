-- Eugenio Culurciello
-- November 2016
-- Deep RNN for reinforcement online learning

if not dqn then require "initenv" end
require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'nngraph'
require 'paths'
local of = require 'opt'
opt = of.parse(arg)
local ut = require 'util'

ut:__init()
-- format options:
opt.pool_frms = 'type=' .. opt.pool_frms_type .. ',size=' .. opt.pool_frms_size

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)

--package.path = '../../catch/?.lua;' .. package.path

os.execute('mkdir '..opt.savedir)
torch.save(paths.concat(opt.savedir,'opt.t7'),opt)

local gameEnv, gameActions = setup(opt) -- setup game environment
print('Game started. Number of game actions:', #gameActions)
local nbStates = opt.gridSize * opt.gridSize
local nbActions = #gameActions
local wi = opt.gridSize
local hi = opt.gridSize
local ch = opt.ch
local batch = opt.batchSize
local hCh = opt.nHidden
local nSeq = 20--4*opt.gridSize -- RNN max sequence length in this game is grid size


-- Params for Stochastic Gradient Descent (our optimizer).
local sgdParams = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}
local predOpt = {
   layers = opt.nLayers, seq = nSeq, height = hi, width = wi, saveGraph = false,
   channels = {ch}, K = nbActions
}
for i = 1, predOpt.layers do
   predOpt.channels[i+1] = 2^(i+3)
end
local prednet = require 'models/prednet'
-- Initialize model generator
prednet:__init(predOpt)
-- Get the model unwrapped over time as well as the prototype
local model, prototype = prednet:getModel()

-- Mean Squared Error for our loss function.
local criterion = nn.MSECriterion()


-- use GPU, if desired:
if opt.useGPU then
   require 'cunn'
   require 'cutorch'
   cutorch.setDevice(opt.gpuId)
   local firstRandInt = torch.random()
   cutorch.manualSeed(firstRandInt) -- set cuda random seed
   model = model:cuda()
   prototype = prototype:cuda()
   criterion = criterion:cuda()
   print('Using GPU number', opt.gpuId)
end

local xb, RNNh0Batch = getBatchInput(
          batch, predOpt.seq, predOpt.height,
          predOpt.width, predOpt.layers, predOpt.channels, 2)
dummyX = {xb, unpack(RNNh0Batch)}
-- test model:
if opt.useGPU then shipToGPU(dummyX) end
print('Dummy')
print(dummyX)
a = model:forward(dummyX)
print('TEST model:', a)

local x, RNNh0Proto = getInput(predOpt.seq, predOpt.height, predOpt.width, predOpt.layers, predOpt.channels, 1)
dummyX = {x, unpack(RNNh0Proto)}
-- test model:
if opt.useGPU then shipToGPU(dummyX) end
print('Dummy proto')
print(dummyX)
a = prototype:forward(dummyX)
print('TEST proto:', a)

-- setup memory and training variables:
local memory = ut:Memory(maxMemory, discount)
local seqMem = torch.Tensor(nSeq, nbStates) -- store sequence of states in successful run
local seqAct = torch.zeros(nSeq, nbActions) -- store sequence of actions in successful run
local epsilon = opt.epsilon -- this will change in the training, so we copy it
local epsUpdate = (epsilon - opt.epsilonMinimumValue)/opt.epochs
local winCount, err, accTime = 0, 0, 0
local randomActions = 0
local action, q
local RNNhProto = {}

print('Begin training:')
for game = 1, opt.epochs do
    sys.tic()
    local steps = 0 -- counts steps to game win

    -- Initialise the environment.
    local screen, reward, gameOver = gameEnv:nextRandomGame()
    local isGameOver = false
    -- The initial state of the environment.
    local currentState = ut:preProcess(screen) -- resize to smaller size
    -- rest RNN to intial state:
    for i=1, #RNNh0Proto do
       if opt.useGPU then
          RNNhProto[i] = RNNh0Proto[i]:cuda():clone()
       else
          RNNhProto[i] = RNNh0Proto[i]:clone()
       end
    end
    while not isGameOver do
        if steps >= nSeq then steps = 0 end -- reset steps if we are still in game
        steps = steps + 1 -- count game steps
        if opt.useGPU then currentState = currentState:cuda() end
        -- print(currentState:view(1, nbStates), RNNhProto)

        q = prototype:forward({
           currentState:view( ch, wi, hi ),
           unpack(RNNhProto)}) -- Forward the current state through the network.

       -- Prepare next iteration state
       RNNhProto = prepareState(q,predOpt)
       if opt.useGPU then shipToGPU(RNNhProto) end
        -- Decides if we should choose a random action,
        -- or an action from the policy network.
        if torch.uniform() < epsilon then
            action = torch.random(1, nbActions)
            randomActions = randomActions + 1
        else
            -- Find the max index (the chosen action).
            local max, index = torch.max(q[1], 1) -- [1] is the output
            action = index[1]
        end
        -- store to memory (but be careful to avoid larger than max seq: nSeq)
        -- store in system memory, not GPU memory!
        if opt.useGPU then currentState = currentState:float() end
        seqMem[steps] = currentState:view( ch, wi, hi)
        seqAct[steps][action] = 1
        screen, reward, gameOver = gameEnv:step(gameActions[action], true)
        local nextState = ut:preProcess(screen)

        if reward >= 1 then
            winCount = winCount + 1
            memory.remember({
                states = seqMem:byte(), -- save as byte, use as float
                actions = seqAct:byte()
            })
            -- We get a batch of training data to train the model.
            local inputs, targets = memory.getBatch(batch, nbActions, nbStates, nSeq, x)
            -- Train the network which returns the error.
            err = err + trainNetwork(model, RNNh0Batch, inputs, targets, criterion, sgdParams, nSeq, nbActions, batch)
        end
        -- Update the current state and if the game is over:
        currentState = nextState
        isGameOver = gameOver

        if opt.display then
            win = image.display({image=currentState:view(opt.gridSize,opt.gridSize), zoom=10, win=win})
            win2 = image.display({image=screen, zoom=1, win=win2})
        end
       collectgarbage()
    end
    seqAct:zero()
    steps = 0 -- resetting step count (which is always grdiSize-2 anyway...)

    if game%opt.progFreq == 0 then
        print(string.format("Game: %d, epsilon: %.2f, error: %.4f, Random Actions: %d, Accuracy: %d%%, time [ms]: %d",
                             game,  epsilon,  err/opt.progFreq, randomActions/opt.progFreq, winCount/opt.progFreq*100, sys.toc()*1000))
        local acc = winCount / opt.progFreq
        ut:write(accTime, acc, err)
        winCount = 0
        err = 0
        randomActions = 0
    end
    if epsilon > opt.epsilonMinimumValue then epsilon = epsilon - epsUpdate  end -- update epsilon for online-learning
    accTime = math.ceil(accTime + sys.toc()*1000)
    collectgarbage()
end
torch.save(opt.savedir.."/model-rnn.net", prototype:clearState())
print("Model saved!")
