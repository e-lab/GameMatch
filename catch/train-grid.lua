-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code

-- playing CATCH version:
-- https://github.com/SeanNaren/QlearningExample.torch

require 'CatchEnvironment'
require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'pl'
lapp = require 'pl.lapp'
opt = lapp [[
  
  Game options:
  --discount            (default 0.9)         discount factor in learning
  --epsilon             (default 1)           initial value of ϵ-greedy action selection
  --epsilonMinimumValue (default 0.001)       final value of ϵ-greedy action selection
  --nbActions           (default 3)           catch number of actions
  
  Training parameters:
  --threads               (default 8)         number of threads used by BLAS routines
  --seed                  (default 1)         initial random seed
  -r,--learningRate       (default 0.1)       learning rate
  -d,--learningRateDecay  (default 1e-9)      learning rate decay
  -w,--weightDecay        (default 0)         L2 penalty on the weights
  -m,--momentum           (default 0.9)       momentum parameter
  --gridSize              (default 10)        state is screen resized to this size 
  --hiddenSize            (default 100)       hidden states in neural net
  --batchSize             (default 50)        batch size for training
  --maxMemory             (default 0.5e3)     Experience Replay buffer memory
  --epoch                 (default 1e3)       number of training steps to perform
  --progFreq              (default 1e2)       frequency of progress output
  --largeSimple                               simple model or not

  Display and save parameters:
  --zoom                  (default 4)        zoom window
  -v, --verbose           (default 2)        verbose output
  --display                                  display stuff
  --savedir          (default './results')   subdirectory to save experiments in
]]

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



-- memory for experience replay:
local function Memory(maxMemory, discount)
    local memory = {}

    -- Appends the experience to the memory.
    function memory.remember(memoryInput)
        table.insert(memory, memoryInput)
        if (#memory > maxMemory) then
            -- Remove the earliest memory to allocate new experience to memory.
            table.remove(memory, 1)
        end
    end

    function memory.getBatch(model, batchSize, nbActions, nbStates)

        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch).
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)

        local inputs = torch.zeros(chosenBatchSize, nbStates)
        local targets = torch.zeros(chosenBatchSize, nbActions)
        --Fill the inputs and targets up.
        for i = 1, chosenBatchSize do
            -- Choose a random memory experience to add to the batch.
            local randomIndex = torch.random(1, memoryLength)
            local memoryInput = memory[randomIndex]

            local target = model:forward(memoryInput.inputState)

            --Gives us Q_sa, the max q for the next state.
            local nextStateMaxQ = torch.max(model:forward(memoryInput.nextState), 1)[1]
            if (memoryInput.gameOver) then
                target[memoryInput.action] = memoryInput.reward
            else
                -- reward + discount(gamma) * max_a' Q(s',a')
                -- We are setting the Q-value for the action to  r + γmax a’ Q(s’, a’). The rest stay the same
                -- to give an error of 0 for those outputs.
                target[memoryInput.action] = memoryInput.reward + discount * nextStateMaxQ
            end
            -- Update the inputs and targets.
            inputs[i] = memoryInput.inputState
            targets[i] = target
        end
        return inputs, targets
    end

    return memory
end

--[[ Runs one gradient update using SGD returning the loss.]] --
local function trainNetwork(model, inputs, targets, criterion, sgdParams)
    local loss = 0
    local x, gradParameters = model:getParameters()
    local function feval(x_new)
        gradParameters:zero()
        local predictions = model:forward(inputs)
        local loss = criterion:forward(predictions, targets)
        local gradOutput = criterion:backward(predictions, targets)
        model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    local _, fs = optim.sgd(feval, x, sgdParams)
    loss = loss + fs[1]
    return loss
end

-- Create the base model.
local model = nn.Sequential()
model:add(nn.Linear(nbStates, hiddenSize))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenSize, hiddenSize))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenSize, nbActions))

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

local env = CatchEnvironment(gridSize)
local memory = Memory(maxMemory, discount)
local epsUpdate = (epsilon - epsilonMinimumValue)/epoch
local winCount = 0

print('Begin training:')
for i = 1, epoch do
    -- Initialise the environment.
    local err = 0
    env.reset()
    local isGameOver = false

    -- The initial state of the environment.
    local currentState = env.observe()

    while not isGameOver do
        local action
        -- Decides if we should choose a random action, or an action from the policy network.
        if torch.random() <= epsilon then
            action = torch.random(1, nbActions)
        else
            -- Forward the current state through the network.
            local q = model:forward(currentState)
            -- Find the max index (the chosen action).
            local max, index = torch.max(q, 1)
            action = index[1]
        end

        local nextState, reward, gameOver = env.act(action)
        if (reward == 1) then winCount = winCount + 1 end
        memory.remember({
            inputState = currentState,
            action = action,
            reward = reward,
            nextState = nextState,
            gameOver = gameOver
        })
        -- Update the current state and if the game is over.
        currentState = nextState
        isGameOver = gameOver

        -- We get a batch of training data to train the model.
        local inputs, targets = memory.getBatch(model, batchSize, nbActions, nbStates)

        -- Train the network which returns the error.
        err = err + trainNetwork(model, inputs, targets, criterion, sgdParams)
    end
        local period = 100
    if i%period == 0 then 
        print(string.format("Epoch %d : epsilon %.2f : err = %f : Win count %d : Accuracy: %.2f", i, epsilon, err, winCount, winCount/period))
        winCount = 0
    end
    -- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
    if epsilon > epsilonMinimumValue then epsilon = epsilon - epsUpdate  end
end
torch.save("catch-model-grid.net", model)
print("Model saved!")
