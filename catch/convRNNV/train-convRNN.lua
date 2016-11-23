-- Eugenio Culurciello
-- Sangpil Kim add convRNN
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
local logger = require 'util'
require 'paths'
local of = require 'opt'
opt = of.parse(arg)
print(opt)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
convRNN = require 'RNNconv'
os.execute('mkdir '..opt.savedir)
torch.save(paths.concat(opt.savedir,'opt.t7'),opt)

logger:__init(opt)
print('Playing Catch game with RNN\n')

local nbStates = opt.gridSize * opt.gridSize
local wi = opt.gridSize
local hi = opt.gridSize
local batch = opt.batchSize
local ch = opt.ch
local hCh = opt.nHidden
local nSeq = opt.gridSize-2 -- RNN sequence length in this game is grid size
local nbActions = opt.nbActions

-- Create the base RNN model:
local model, prototype
local RNNh0Batch = {} -- initial state
local RNNh0Proto = {} -- initial state - prototype
local RNNhProto = {} -- state to loop through prototype in inference

if opt.fw then
  print('Created fast-weights RNN with:\n- input size:', nbStates, '\n- number hidden:', opt.nHidden,
    '\n- layers:', opt.nLayers, '\n- output size:', nbActions, '\n- sequence length:', nSeq,
    '\n- fast weights states:', opt.nFW)
  model, prototype = convRNN.getModel(ch, hCh, opt.nLayers, nbActions, nSeq, batch, wi, hi, opt.nFW)
else
  print('Created RNN with:\n- input size:', nbStates, '\n- number hidden:', opt.nHidden,
      '\n- layers:', opt.nLayers, '\n- output size:', nbActions, '\n- sequence lenght:',  nSeq)
  model, prototype = convRNN.getModel(ch, hCh, opt.nLayers, nbActions, nSeq, batch, wi, hi)
end

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

-- Default RNN intial state set to zero:
for l = 1, opt.nLayers do
   RNNh0Batch[l] = torch.zeros(batch, hCh, wi, hi)
   RNNh0Proto[l] = torch.zeros(hCh, wi, hi) -- prototype forward does not work on batches
   if opt.useGPU then RNNh0Batch[l]=RNNh0Batch[l]:cuda() RNNh0Proto[l]=RNNh0Proto[l]:cuda() end
end

-- Params for Stochastic Gradient Descent (our optimizer).
local sgdParams = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}

local ttest = {torch.Tensor(batch, nSeq, ch, wi, hi),
         torch.Tensor(batch, hCh, wi, hi)}
-- test model:
if opt.useGPU then
   for i = 1, #ttest do
      ttest[i] = ttest[i]:cuda()
   end
end
print(ttest)
local a = model:forward(ttest)
print('TEST model:', a)

local gameEnv = CatchEnvironment(opt.gridSize) -- init game engine
local memory = Memory(maxMemory, discount)
local seqMem = torch.zeros(nSeq, nbStates) -- store sequence of states in successful run
local seqAct = torch.zeros(nSeq, nbActions) -- store sequence of actions in successful run
local epsilon = opt.epsilon -- this will change in the training, so we copy it
local epsUpdate = (epsilon - opt.epsilonMinimumValue)/opt.epochs
local winCount, err, accTime = 0, 0, 0
local randomActions = 0

print('Begin training:')
for game = 1, opt.epochs do
    sys.tic()
    local steps = 0 -- counts steps to game win

    -- Initialise the environment.
    gameEnv.reset()
    local isGameOver = false

    -- The initial state of the environment.
    local currentState = gameEnv.observe()

    -- rest RNN to intial state:
    RNNhProto = table.unpack(RNNh0Proto)

    while not isGameOver do
        steps = steps + 1 -- count game steps
        local action, q
        if opt.useGPU then currentState = currentState:cuda() end
        -- print(currentState:view(1, nbStates), RNNhProto)
        q = prototype:forward({currentState:view( ch, wi, hi ), RNNhProto}) -- Forward the current state through the network.
        RNNhProto = q[1]
        -- Decides if we should choose a random action, or an action from the policy network.
        if torch.uniform() < epsilon then
            action = torch.random(1, nbActions)
            randomActions = randomActions + 1
        else
            -- Find the max index (the chosen action).
            local max, index = torch.max(q[2], 1) -- [2] is the output, [1] is state...
            action = index[1]
        end
        -- store to memory
        if opt.useGPU then currentState = currentState:float() end -- store in system memory, not GPU memory!
        seqMem[steps] = currentState -- store state sequence into memory
        seqAct[steps][action] = 1

        local nextState, reward, gameOver = gameEnv.act(action)

        if (reward == 1) then
            winCount = winCount + 1
            memory.remember({
                states = seqMem:byte(), -- save as byte, use as float
                actions = seqAct:byte()
            })
            -- We get a batch of training data to train the model.
            local inputs, targets = memory.getBatch(batch, nbActions, nbStates, nSeq, ttest[1])
            -- Train the network which returns the error.
            err = err + trainNetwork(model, RNNh0Batch, inputs, targets, criterion, sgdParams, nSeq, nbActions, batch)
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
        logger:write(acc)
        logger:tbwrite(accTime, acc)
        winCount = 0
        err = 0
        randomActions = 0
    end
    if epsilon > opt.epsilonMinimumValue then epsilon = epsilon - epsUpdate  end -- update epsilon for online-learning
    accTime = math.ceil(accTime + sys.toc()*1000)
    collectgarbage()
end
torch.save(opt.savedir.."/catch-model-convRNN.net", prototype:clearState())
print("Model saved!")
