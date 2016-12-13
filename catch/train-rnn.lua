-- Eugenio Culurciello
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

require 'pl'
lapp = require 'pl.lapp'
opt = lapp [[

  Game options:
  --gridSize            (default 10)          game grid size
  --discount            (default 0.9)         discount factor in learning
  --epsilon             (default 1)           initial value of ϵ-greedy action selection
  --epsilonMinimumValue (default 0.02)        final value of ϵ-greedy action selection
  --nbActions           (default 3)           catch number of actions
  --playFile            (default '')          human play file to initialize exp. replay memory

  Training parameters:
  --threads               (default 8)         number of threads used by BLAS routines
  --seed                  (default 1)         initial random seed
  -r,--learningRate       (default 2e-3)      learning rate
  -d,--learningRateDecay  (default 1e-9)      learning rate decay
  -w,--weightDecay        (default 0)         L2 penalty on the weights
  -m,--momentum           (default 0.9)       momentum parameter
  --batchSize             (default 64)        batch size for training
  --maxMemory             (default 1e3)       Experience Replay buffer memory
  --epochs                (default 1e4)       number of training steps to perform
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

local rnn = require 'RNN'

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)
print('Playing Catch game with RNN\n')

local nbStates = opt.gridSize * opt.gridSize
local nSeq = opt.gridSize-2 -- RNN sequence length in this game is grid size
local nbActions = opt.nbActions

-- memory for experience replay:
local memory = {}
local function initMemory(maxMemory)
    memory.states = torch.zeros(maxMemory, nSeq, nbStates)
    memory.actions = torch.ones(maxMemory, nSeq)
    memory.capacity = maxMemory
    memory.size = 0
    memory.pos = 1
   
    -- batch buffers:
    local binputs = torch.zeros(opt.batchSize, nSeq, nbStates)
    local btargets = torch.ones(opt.batchSize, nSeq)

    if opt.playFile ~= '' then
        memory = torch.load(opt.playFile)
        print('Loaded experience replay memory with play file:', opt.playFile)
    else
        print('Initialized empty experience replay memory')
    end

    -- Appends the experience to the memory.
    function memory.remember(seqs,acts)
      memory.states[memory.pos] = seqs:clone()
      memory.actions[memory.pos] = acts:clone()
       
      memory.pos = (memory.pos + 1) % memory.capacity
      if memory.pos == 0 then memory.pos = 1 end -- to prevent issues!
      memory.size = math.min(memory.size + 1, memory.capacity)
    end

    function memory.getBatch(batchSize)
        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch)
        local chosenBatchSize = math.min(opt.batchSize, memory.size)

        -- create inputs and targets:
        local ri = torch.randperm(memory.size)
        for i = 1, chosenBatchSize do
            binputs[i] = memory.states[ri[i]]
            btargets[i]= memory.actions[ri[i]]
        end
        if opt.useGPU then binputs = binputs:cuda() btargets = btargets:cuda() end

        return binputs, btargets
    end
end

-- training code:
local function trainNetwork(model, state, inputs, targets, criterion, sgdParams)
    local loss = 0
    local x, gradParameters = model:getParameters()

    local function feval(x_new)
      local loss = 0
      local grOut = {}
      -- print(state)
      -- print(targets)
      -- print(inputs)
      -- io.read()
      local inputs = { inputs, table.unpack(state) } -- attach RNN states to input
      local out = model:forward(inputs)
      -- process each sequence step at a time:
      for i = 1, nSeq do
          gradParameters:zero()
          loss = loss + criterion:forward(out[i], targets[{{},i}])
          grOut[i] = criterion:backward(out[i], targets[{{},i}])
      end
      table.insert(grOut, state) -- attach RNN states to grad output
      model:backward(inputs, grOut)
      -- print(loss)
      return loss, gradParameters
  end

    local _, fs = optim.rmsprop(feval, x, sgdParams)

    loss = loss + fs[1]
    return loss
end

-- Create the base RNN model:
local model, prototype
local RNNh0Batch = {} -- initial state
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
local criterion = nn.ClassNLLCriterion()--nn.MSECriterion()

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

-- Params for Stochastic Gradient Descent (our optimizer).
local sgdParams = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}


-- test model:
print('Testing model and prototype RNN:')
local ttest
if opt.useGPU then ttest = {torch.CudaTensor(1, nbStates), torch.CudaTensor(1, opt.nHidden)}
else ttest = {torch.Tensor(1, nbStates), torch.Tensor(1, opt.nHidden)} end
print(ttest)
local a = prototype:forward(ttest)
print('TEST prototype:', a)
if opt.useGPU then ttest = {torch.CudaTensor(opt.batchSize, nSeq, nbStates), torch.CudaTensor(opt.batchSize, opt.nHidden)}
else ttest = {torch.Tensor(opt.batchSize, nSeq, nbStates), torch.Tensor(opt.batchSize, opt.nHidden)} end
print(ttest)
local a = model:forward(ttest)
print('TEST model:', a)


local gameEnv = CatchEnvironment(opt.gridSize) -- init game engine
initMemory(opt.maxMemory) -- init memory
local seqMem = torch.zeros(nSeq, nbStates) -- store sequence of states in successful run
local seqAct = torch.ones(nSeq) -- store sequence of actions in successful run
local epsilon = opt.epsilon -- this will change in the training, so we copy it
local epsUpdate = (epsilon - opt.epsilonMinimumValue)/opt.epochs
local winCount = 0
local err = 0
local randomActions = 0
local accTime = 0

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
        q = prototype:forward({currentState:view(1, nbStates), RNNhProto}) -- Forward the current state through the network.
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
        -- store to memory
        if opt.useGPU then currentState = currentState:float() end -- store in system memory, not GPU memory!
        seqMem[steps] = currentState:clone() -- store state sequence into memory
        seqAct[steps] = action

        local nextState, reward, gameOver = gameEnv.act(action)

        if reward >= 1 then
            winCount = winCount + 1
            memory.remember(seqMem, seqAct)
            -- We get a batch of training data to train the model.
            local inputs, targets = memory.getBatch(opt.batchSize, nbActions, nbStates)
            -- Train the network which returns the error.
            err = err + trainNetwork(model, RNNh0Batch, inputs, targets, criterion, sgdParams)
        end
        -- Update the current state and if the game is over.
        currentState = nextState
        isGameOver = gameOver

        if opt.display then
            win = image.display({image=currentState:view(opt.gridSize,opt.gridSize), zoom=10, win=win})
        end
    end
    seqAct:fill(1)
    steps = 0 -- resetting step count (which is always grdiSize-2 anyway...)

    if game%opt.progFreq == 0 then
        print(string.format("Game: %d, epsilon: %.2f, error: %.4f, Random Actions: %d, Accuracy: %d%%, time [ms]: %d",
                             game,  epsilon,  err/opt.progFreq, randomActions/opt.progFreq, winCount/opt.progFreq*100, sys.toc()*1000))
        local acc = winCount / opt.progFreq
        winCount = 0
        err = 0
        randomActions = 0
    end
    if epsilon > opt.epsilonMinimumValue then epsilon = epsilon - epsUpdate  end -- update epsilon for online-learning
    accTime = math.ceil(accTime + sys.toc()*1000)
    collectgarbage()
end
torch.save(opt.savedir.."/catch-proto-rnn.net", prototype:float():clearState())
print("Prototype model saved!")
