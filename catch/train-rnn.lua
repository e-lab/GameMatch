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
  --epsilonMinimumValue (default 0.001)       final value of ϵ-greedy action selection
  --nbActions           (default 3)           catch number of actions
  --playFile            (default '')          human play file to initialize exp. replay memory

  Training parameters:
  --threads               (default 8)         number of threads used by BLAS routines
  --seed                  (default 1)         initial random seed
  -r,--learningRate       (default 0.1)       learning rate
  -d,--learningRateDecay  (default 1e-9)      learning rate decay
  -w,--weightDecay        (default 0)         L2 penalty on the weights
  -m,--momentum           (default 0.9)       momentum parameter
  --batchSize             (default 1)         batch size for training
  --maxMemory             (default 1e3)       Experience Replay buffer memory
  --epoch                 (default 1e5)       number of training steps to perform
  
  Model parameters:
  --fw                                        Use FastWeights or not
  --nLayers               (default 1)         RNN layers
  --nHidden               (default 128)       RNN hidden size
  --nFW                   (default 4)         number of fast weights previous vectors

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
-- os.execute('mkdir '..opt.savedir)


local epsilon = opt.epsilon
local epsilonMinimumValue = opt.epsilonMinimumValue
local nbActions = opt.nbActions
local epoch = opt.epoch
local hiddenSize = opt.hiddenSize
local maxMemory = opt.maxMemory
local batchSize = opt.batchSize
local gridSize = opt.gridSize
local nbStates = gridSize * gridSize
local discount = opt.discount
local nSeq = gridSize-2 -- RNN sequence length in this game is grid size


-- memory for experience replay:
local function Memory(maxMemory, discount)
    local memory

    if opt.playFile ~= '' then
        memory = torch.load(opt.playFile)
        print('Loaded experience replay memory with play file:', opt.playFile)
    else
        memory = {}
        print('Initialized empty experience replay memory')
    end
    
    -- Appends the experience to the memory.
    function memory.remember(memoryInput)
        table.insert(memory, memoryInput)
        if (#memory > maxMemory) then
            -- Remove the earliest memory to allocate new experience to memory.
            table.remove(memory, 1)
        end
    end

    function memory.getBatch(batchSize, nbActions, nbStates)
        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch)
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)

        local inputs = torch.Tensor(chosenBatchSize, nSeq, nbStates)
        local targets = torch.zeros(chosenBatchSize, nSeq, nbActions)

        -- create inputs and targets:
        for i = 1, chosenBatchSize do
            local randomIndex = torch.random(1, memoryLength)
            inputs[i] = memory[randomIndex].states:float() -- save as byte, use as float
            targets[i]= memory[randomIndex].actions:float()
        end

        return inputs, targets
    end

    return memory
end

-- Converts input tensor into table of dimension equal to first dimension of input tensor
-- and adds padding of zeros, which in this case are states
local function tensor2Table(inputTensor, padding)
   local outputTable = {}
   for t = 1, inputTensor:size(1) do outputTable[t] = inputTensor[t] end
   for l = 1, padding do outputTable[l + inputTensor:size(1)] = h0[l]:clone() end
   return outputTable
end

-- training code:
local function trainNetwork(model, state, inputs, targets, criterion, sgdParams)
    local loss = 0
    local x, gradParameters = model:getParameters()
    local function feval(x_new)
        gradParameters:zero()
        local out = model:forward( {inputs:squeeze(), table.unpack(state)} )
        local predictions = torch.Tensor(nSeq, nbActions)
        for i = 1, nSeq do
            predictions[i] = out[i]
        end
        local loss = criterion:forward(predictions, targets:squeeze())
        local grOut = criterion:backward(predictions, targets)
        local gradOutput = tensor2Table(grOut,0)
        gradOutput[#gradOutput+1] = state[1]
        model:backward(inputs:squeeze(), gradOutput)
        return loss, gradParameters
    end

    local _, fs = optim.sgd(feval, x, sgdParams)
    loss = loss + fs[1]
    return loss
end

-- Create the base RNN model:
local model, prototype
local RNNh0 = {} -- initial state
local RNNh = {} -- state to loop through prototype in inference

if opt.fw then
  print('Created fast-weights RNN with:\n- input size:', nbStates, '\n- number hidden:', opt.nHidden, 
    '\n- layers:', opt.nLayers, '\n- output size:',  nbActions, '\n- sequence length:', nSeq, 
    '\n- fast weights states:', opt.nFW)
  model, prototype = rnn.getModel(nbStates, opt.nHidden, opt.nLayers, nbActions, nSeq, opt.nFW)
else
  print('Created RNN with:\n- input size:', nbStates, '\n- number hidden:', opt.nHidden, 
      '\n- layers:', opt.nLayers, '\n- output size:',  nbActions, '\n- sequence lenght:',  nSeq)
  model, prototype = rnn.getModel(nbStates, opt.nHidden, opt.nLayers, nbActions, nSeq)
end

-- Default RNN intial state set to zero:
for l = 1, opt.nLayers do
   RNNh0[l] = torch.zeros(opt.nHidden)
   RNNh[l] = torch.zeros(opt.nHidden)
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

-- Mean Squared Error for our loss function.
local criterion = nn.MSECriterion()

local gameEnv = CatchEnvironment(gridSize)
local memory = Memory(maxMemory, discount)
local seqMem = torch.Tensor(nSeq, nbStates) -- store sequence of states in successful run
local seqAct = torch.zeros(nSeq, nbActions) -- store sequence of actions in successful run
local epsUpdate = (epsilon - epsilonMinimumValue)/epoch
local winCount = 0

print('Begin training:')
for game = 1, epoch do
    sys.tic()
    -- Initialise the environment.
    local err = 0
    local steps = 0 -- counts steps to game win
    gameEnv.reset()
    local isGameOver = false

    -- The initial state of the environment.
    local currentState = gameEnv.observe()

    -- rest RNN to intial state:
    RNNh = table.unpack(RNNh0)

    while not isGameOver do
        steps = steps+1
        local action, q
        q = prototype:forward({currentState, RNNh}) -- Forward the current state through the network.
        RNNh = q[1]
        -- Decides if we should choose a random action, or an action from the policy network.
        if torch.uniform() < epsilon then
            action = torch.random(1, nbActions)
            -- print(action)
        else
            -- Find the max index (the chosen action).
            local max, index = torch.max(q[2], 1) -- [2] is the output, [1] is state...
            action = index[1]
        end
        -- store to memory
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
            local inputs, targets = memory.getBatch(batchSize, nbActions, nbStates)
            -- Train the network which returns the error.
            err = err + trainNetwork(model, RNNh0, inputs, targets, criterion, sgdParams)
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
        print(string.format("Game %d, epsilon %.2f, err = %.4f, Win count %d, Accuracy: %.2f, time [ms]: %d", 
                             game,    epsilon,      err,        winCount,     winCount/opt.progFreq, sys.toc()*1000))
        winCount = 0
    end
    -- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
    if epsilon > epsilonMinimumValue then epsilon = epsilon - epsUpdate  end
end
torch.save("catch-model-rnn.net", prototype:clearState())
print("Model saved!")