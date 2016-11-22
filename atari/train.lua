-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code
-- an implementation of: http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
-- inspired by: http://outlace.com/Reinforcement-Learning-Part-3/
-- or: https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html
--
-- new RNN version!

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

local gameEnv, gameActions, agent, opt = setup(opt) -- setup game environment
print('Game started. Number of game actions:', #gameActions)
local nbActions = #gameActions
local nbStates = opt.gridSize * opt.gridSize
local nSeq = 4*opt.gridSize -- RNN max sequence length in this game is grid size

-- memory for experience replay:
local function Memory(maxMemory, discount)
    local memory

    if opt.playFile ~= '' then
        memory = torch.load(opt.playFile)
        print('Loaded experience replay memory with play file:', opt.playFile)
        opt.maxMemory = #memory -- resize this to loaded file
    else
        memory = {}
        print('Initialized empty experience replay memory')
    end
    
    -- Appends the experience to the memory.
    function memory.remember(memoryInput)
        table.insert(memory, memoryInput)
        if (#memory > opt.maxMemory) then
            -- Remove the earliest memory to allocate new experience to memory.
            table.remove(memory, 1)
        end
    end

    function memory.getBatch(batchSize, nbActions, nbStates)
        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch)
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)

        local inputs = torch.zeros(batchSize, nSeq, nbStates)
        local targets = torch.zeros(batchSize, nSeq, nbActions)

        -- create inputs and targets:
        for i = 1, chosenBatchSize do
            local randomIndex = torch.random(1, memoryLength)
            inputs[i] = memory[randomIndex].states:float() -- save as byte, use as float
            targets[i]= memory[randomIndex].actions:float()
        end
        if opt.useGPU then inputs = inputs:cuda() targets = targets:cuda() end

        return inputs, targets
    end

    return memory
end

-- Converts input tensor into table of dimension equal to first dimension of input tensor
-- and adds padding of zeros, which in this case are states
local function tensor2Table(inputTensor, padding, state)
   local outputTable = {}
   for t = 1, inputTensor:size(1) do outputTable[t] = inputTensor[t] end
   for l = 1, padding do outputTable[l + inputTensor:size(1)] = state[l]:clone() end
   return outputTable
end

-- training code:
local function trainNetwork(model, state, inputs, targets, criterion, sgdParams)
    local loss = 0
    local x, gradParameters = model:getParameters()
    
    local function feval(x_new)
        gradParameters:zero()
        inputs = {inputs, table.unpack(state)} -- attach states
        local out = model:forward(inputs)
        local predictions = torch.Tensor(nSeq, opt.batchSize, nbActions)
        if opt.useGPU then predictions = predictions:cuda() end
        -- create table of outputs:
        for i = 1, nSeq do
            predictions[i] = out[i]
        end
        predictions = predictions:transpose(2,1)
        -- print('in', inputs) print('outs:', out) print('targets', {targets}) print('predictions', {predictions})
        local loss = criterion:forward(predictions, targets)
        local grOut = criterion:backward(predictions, targets)
        grOut = grOut:transpose(2,1)
        local gradOutput = tensor2Table(grOut, 1, state)
        model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    local _, fs = optim.sgd(feval, x, sgdParams)
    
    loss = loss + fs[1]
    return loss
end

function preProcess(im)
  local net = nn.SpatialMaxPooling(4,4,4,4)
  -- local pooled = net:forward(im[1])
  local pooled = net:forward(im[1][{{},{94,194},{9,152}}])
  -- print(pooled:size())
  local out = image.scale(pooled, opt.gridSize, opt.gridSize):sum(1):div(3)
  return out
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
local memory = Memory(maxMemory, discount)
local seqMem = torch.Tensor(nSeq, nbStates) -- store sequence of states in successful run
local seqAct = torch.zeros(nSeq, nbActions) -- store sequence of actions in successful run
local epsilon = opt.epsilon -- this will change in the training, so we copy it
local epsUpdate = (epsilon - opt.epsilonMinimumValue)/opt.epochs
local winCount = 0
local err = 0
local randomActions = 0

print('Begin training:')
for game = 1, opt.epochs do
    sys.tic()
    local steps = 0 -- counts steps to game win
    
    -- Initialise the environment.
    local screen, reward, gameOver = gameEnv:nextRandomGame()
    local currentState = preProcess(screen) -- resize to smaller size
    local isGameOver = false
    -- reset RNN to intial state:
    RNNhProto = table.unpack(RNNh0Proto)
    
    while not isGameOver do
        if steps >= nSeq then steps = 0 end -- reset steps if we are still in game
        steps = steps + 1 -- count game steps
        local action
        if opt.useGPU then currentState = currentState:cuda() end
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
        local nextState = preProcess(screen)

        if (reward == 1) then 
            winCount = winCount + 1 
            memory.remember({
                states = seqMem:byte(), -- save as byte, use as float
                actions = seqAct:byte()
            })
            -- We get a batch of training data to train the model.
            local inputs, targets = memory.getBatch(opt.batchSize, nbActions, nbStates)
            -- Train the network which returns the error.
            err = err + trainNetwork(model, RNNh0Batch, inputs, targets, criterion, sgdParams)
            break
        end
        -- Update the current state and if the game is over.
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
        winCount = 0
        err = 0 
        randomActions = 0
    end
    if epsilon > opt.epsilonMinimumValue then epsilon = epsilon - epsUpdate  end -- update epsilon for online-learning
end
torch.save(opt.savedir.."/model-rnn.net", prototype:clearState())
print("Model saved!")