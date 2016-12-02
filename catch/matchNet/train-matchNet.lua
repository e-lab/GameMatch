-- Eugenio Culurciello
-- Sangpil Kim add matchNet
-- October 2016
-- learning with RNN

-- playing CATCH version:
-- loosely based on: https://github.com/SeanNaren/QlearningExample.torch

require 'CatchEnvironment'
require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'nngraph'
require 'paths'
local logger = require 'util'
local of = require 'opt'
opt = of.parse(arg)
print(opt)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
local matchNet = require 'models/prednet'
os.execute('mkdir '..opt.savedir)
torch.save(paths.concat(opt.savedir,'opt.t7'),opt)

logger:__init(opt)
print('Playing Catch game with RNN\n')

local nbStates = opt.gridSize * opt.gridSize
local nbActions = opt.nbActions
local wi = opt.gridSize
local hi = opt.gridSize
local ch = opt.ch
local batch = opt.batchSize
local hCh = opt.nHidden
--opt.gridSize-2 -- RNN sequence length in this game is grid size
local nSeq = opt.gridSize - 2

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
for l = 2, opt.nLayers + 1 do
   predOpt.channels[l] = 2^(l +  3)
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



local xb, RNNh0Batch = getBatchInput(batch, predOpt.seq, predOpt.height, predOpt.width, predOpt.layers, predOpt.channels, 2)
dummyX = {xb, unpack(RNNh0Batch)}
-- test model:
if opt.useGPU then
   for i = 1, #dummyX do
      dummyX[i] = dummyX[i]:cuda()
   end
end
print('Dummy')
print(dummyX)
a = model:forward(dummyX)
print('TEST model:', a)

local x, RNNh0Proto = getInput(predOpt.seq, predOpt.height, predOpt.width, predOpt.layers, predOpt.channels, 1)
dummyX = {x, unpack(RNNh0Proto)}
-- test model:
if opt.useGPU then
   for i = 1, #dummyX do
      dummyX[i] = dummyX[i]:cuda()
   end
end
print('Dummy proto')
a = prototype:forward(dummyX)
print('TEST proto:', a)

local gameEnv = CatchEnvironment(opt.gridSize) -- init game engine
local memory = Memory(maxMemory, discount)
local seqMem = torch.zeros(nSeq, ch, wi, hi) -- store sequence of states in successful run
local seqAct = torch.zeros(nSeq, nbActions) -- store sequence of actions in successful run
local epsilon = opt.epsilon -- this will change in the training, so we copy it
local epsUpdate = (epsilon - opt.epsilonMinimumValue)/opt.epochs
local winCount, err, accTime = 0, 0, 0
local randomActions = 0
local RNNhProto = {}

print('Begin training:')
for game = 1, opt.epochs do
    sys.tic()
    local steps = 0 -- counts steps to game win

    -- Initialise the environment.
    gameEnv.reset()
    local reward = 0
    local isGameOver = false
    local nextState

    -- The initial state of the environment.
    local currentState = gameEnv.observe()

    -- rest RNN to intial state:
    for i=1, #RNNh0Proto do
          RNNhProto[i] = RNNh0Proto[i]:clone()
    end
    -- Ship to GPU
    if opt.useGPU then tableGPU(RNNhProto) end
    while not isGameOver do
        if steps >= nSeq then steps = 0 end -- reset steps if we are still in game
        steps = steps + 1 -- count game steps
        local action, q
        if opt.useGPU then currentState = currentState:cuda() end
        q = prototype:forward({currentState:view( ch, wi, hi ), unpack(RNNhProto)}) -- Forward the current state through the network.
       for i =1 , #q do
          if i == 1 then
             -- This is defined in the model code
             RNNhProto[i] = torch.zeros(
               predOpt.channels[predOpt.layers+1],
               predOpt.height/2^(opt.nLayers), predOpt.width/2^(opt.nLayers))
          else
             RNNhProto[i] = q[i]:clone()
          end
       end
       if opt.useGPU then
          for i = 1, #RNNhProto do
             RNNhProto[i] = RNNhProto[i]:cuda()
          end
       end
        -- Decides if we should choose a random action, or an action from the policy network.
        if torch.uniform() < epsilon then
            action = torch.random(1, nbActions)
            randomActions = randomActions + 1
        else
            -- Find the max index (the chosen action).
            -- q[1] is the action output
            -- Select maximum to get index
            local max, index = torch.max(q[1], 1)
            action = index[1]
        end
        -- store to memory
        if opt.useGPU then currentState = currentState:float() end -- store in system memory, not GPU memory!
        seqMem[steps] = currentState:view( ch, wi, hi) -- store state sequence into memory
        seqAct[steps][action] = 1

        nextState, reward, gameOver = gameEnv.act(action)

        -- Select action from trajectory that maximize reward
        if (reward >= 1) then
            winCount = winCount + 1
            memory.remember({
                states = seqMem:byte(), -- save as byte, use as float
                actions = seqAct:byte()
            })
            local mlength = memory.getLengty()
            if mlength > batch then
               -- We get a batch of training data to train the model.
               local inputs, targets, RNNhBatch = memory.getBatch(
                  batch, nbActions, nbStates, nSeq, x, predOpt)
               -- Train the network which returns the error.
               err = err + trainNetwork(model, RNNhBatch, inputs, targets, criterion, sgdParams, nSeq, nbActions, batch)
            end
        end
        -- Update the current state and if the game is over.
        currentState = nextState
        isGameOver = gameOver

        if opt.display then
            win = image.display({image=currentState:view(opt.gridSize,opt.gridSize), zoom=10, win=win})
        end
    end
    seqAct:zero()
    steps = 0 -- resetting step count (which is always grdiSize-2 anyway...)

    if game%opt.progFreq == 0 then
        print(string.format("Game: %d, epsilon: %.2f, error: %.4f, Random Actions: %d, Accuracy: %d%%, time [ms]: %d",
                             game,  epsilon,  err/opt.progFreq, randomActions/opt.progFreq, winCount/opt.progFreq*100, sys.toc()*1000))
        local acc = winCount / opt.progFreq
        logger:write(accTime, acc, err)
        winCount = 0
        err = 0
        randomActions = 0
    end
    if epsilon > opt.epsilonMinimumValue then epsilon = epsilon - epsUpdate  end -- update epsilon for online-learning
    accTime = math.ceil(accTime + sys.toc()*1000)
    collectgarbage()
end
torch.save(opt.savedir.."/catch-model-matchNet.net", prototype:clearState())
print("Model saved!")
