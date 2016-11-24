-- Eugenio Culurciello
-- November 2016
-- Deep RNN for reinforcement online learning

if not dqn then
   require "initenv"
end
require 'image'
require 'optim'
local of = require 'opt'
opt = of.parse(arg)
local ut = require 'util'

--Create logger inside init
ut:__init()
-- format options:
opt.pool_frms = 'type=' .. opt.pool_frms_type .. ',size=' .. opt.pool_frms_size

if opt.verbose >= 1 then
    print('Using options:')
    for k, v in pairs(opt) do
        print(k, v)
    end
end

package.path = '../catch/?.lua;' .. package.path
local rnn = require 'RNN'

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)

local gameEnv, gameActions = setup(opt) -- setup game environment
print('Game started. Number of game actions:', #gameActions)
local nbActions = #gameActions
local nbStates = opt.gridSize * opt.gridSize
local nSeq = 4*opt.gridSize -- RNN max sequence length in this game is grid size


-- Params for Stochastic Gradient Descent (our optimizer).
local sgdParams = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}

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
local memory = ut:Memory(maxMemory, discount)
local seqMem = torch.Tensor(nSeq, nbStates) -- store sequence of states in successful run
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
    local screen, reward, gameOver = gameEnv:nextRandomGame()
    local currentState = ut:preProcess(screen) -- resize to smaller size
    local isGameOver = false
    -- reset RNN to intial state:
    RNNhProto = table.unpack(RNNh0Proto)
    while not isGameOver do
        if steps >= nSeq then steps = 0 end -- reset steps if we are still in game
        steps = steps + 1 -- count game steps
        local action
        if opt.useGPU then currentState = currentState:cuda() end
        prototype:forward({currentState:view(1, nbStates), table.unpack(RNNh0Proto)}) -- HAVE TO DO THIS to clear proto state, or crashes
        local q = prototype:forward({currentState:view(1, nbStates), RNNhProto}) -- Forward the current state through the network.
        RNNhProto = q[1]
        -- Decides if we should choose a random action, or an action from the policy network.
        if torch.uniform() < epsilon then
            action = torch.random(1, nbActions)
            randomActions = randomActions + 1
        else
            -- Find the max index (the chosen action).
            local max, index = torch.max(q[2][1], 1) -- [2] is the output, [1] is state...
            action = index[1]
        end
        -- store to memory (but be careful to avoid larger than max seq: nSeq)
        if opt.useGPU then currentState = currentState:float() end -- store in system memory, not GPU memory!
        seqMem[steps] = currentState -- store state sequence into memory
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
            local inputs, targets = memory.getBatch(opt.batchSize, nbActions, nbStates, nSeq)
            -- Train the network which returns the error.
            err = err + ut:trainNetwork(model, RNNh0Batch, inputs, targets, criterion, sgdParams, nSeq, nbActions)
        end
        -- Update the current state and if the game is over:
        currentState = nextState
        isGameOver = gameOver

        if opt.display then
            win = image.display({image=currentState:view(opt.gridSize,opt.gridSize), zoom=10, win=win})
            win2 = image.display({image=screen, zoom=1, win=win2})
        end
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
