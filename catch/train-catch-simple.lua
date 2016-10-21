-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code
-- an implementation of: http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
-- inspired by: http://outlace.com/Reinforcement-Learning-Part-3/
-- or: https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

-- playing CATCH version:
-- https://github.com/Kaixhin/rlenvs

-- this version 

-- if not dqn then
    -- require "initenv"
-- end

local image = require 'image'
local Catch = require 'rlenvs/Catch' -- install: https://github.com/Kaixhin/rlenvs
require 'torch'
require 'nn'
require 'nngraph'
require 'image'
require 'optim'

require 'pl'
lapp = require 'pl.lapp'
opt = lapp [[

  Game options:
  --framework           (default 'alewrap')         name of training framework
  --env                 (default 'breakout')        name of environment to use')
  --game_path           (default 'roms/')           path to environment file (ROM)
  --env_params          (default 'useRGB=true')     string of environment parameters
  --pool_frms_type      (default 'max')             pool inputs frames mode
  --pool_frms_size      (default '1')               pool inputs frames size
  --actrep              (default 1)                 how many times to repeat action
  --randomStarts        (default 30)                play action 0 between 1 and random_starts number of times at the start of each training episode
  --gamma               (default 0.99)              discount factor in learning
  --epsilon             (default 1)                 initial value of ϵ-greedy action selection
  
  Training parameters:
  --threads               (default 8)         number of threads used by BLAS routines
  --seed                  (default 1)         initial random seed
  -r,--learningRate       (default 0.001)     learning rate
  -d,--learningRateDecay  (default 0)         learning rate decay
  -w,--weightDecay        (default 0)         L2 penalty on the weights
  -m,--momentum           (default 0.9)       momentum parameter
  --imSize                (default 24)        state is screen resized to this size 
  --batchSize             (default 32)        batch size for training
  --ERBufSize             (default 1e3)       Experience Replay buffer memory
  --sFrames               (default 4)         input frames to stack as input / learn every update_freq steps of game
  --steps                 (default 1e4)       number of training steps to perform
  --progFreq              (default 1e2)       frequency of progress output
  --testFreq              (default 1e9)       frequency of testing
  --evalSteps             (default 1e4)       number of test games to play to test results
  --useGPU                                    use GPU in training
  --gpuId                 (default 1)         which GPU to use

  Display and save parameters:
  --zoom                  (default 4)     zoom window
  -v, --verbose           (default 2)     verbose output
  --display                               display stuff
  --savedir      (default './results')    subdirectory to save experiments in
]]

-- format options:
-- opt.pool_frms = 'type=' .. opt.pool_frms_type .. ',size=' .. opt.pool_frms_size
-- opt.saveFreq = opt.steps / 10 -- save 10 times in total

-- if opt.verbose >= 1 then
    -- print('Using options:')
    -- for k, v in pairs(opt) do
        -- print(k, v)
    -- end
-- end

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)

-- Clamps a number to within a certain range.
function math.clamp(n, low, high) return math.min(math.max(low, n), high) end


--[[ The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
--   and creates batches of experiences based on the mini-batch size for training.]] --
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

    function memory.getBatch(model, batchSize, nbActions, gridSize)

        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch).
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)

        local inputs = torch.zeros(chosenBatchSize, 1, gridSize, gridSize)
        local targets = torch.zeros(chosenBatchSize, nbActions)

        --Fill the inputs and targets up.
        for i = 1, chosenBatchSize do
            -- Choose a random memory experience to add to the batch.
            local randomIndex = math.random(1, memoryLength)
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


--- General setup:
-- local gameEnv, gameActions, agent, opt = setup(opt)
local gameEnv = Catch({level = 2})
local stateSpec = gameEnv:getStateSpec()
local actionSpec = gameEnv:getActionSpec()
local observation = gameEnv:start()
print('screen size is:', observation:size())
-- print(stateSpec,actionSpec)
gameActions = {0,1,2} -- game actions from CATCH
-- print(gameActions, #gameActions)

-- set parameters and vars:
local epsilon = opt.epsilon -- ϵ-greedy action selection
local gamma = opt.gamma -- discount factor
local err = 0 -- loss function error (average over opt.progFreq steps)
local w, dE_dw
local optimState = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
  weightDecay = opt.weightDecay
}
local totalReward = 0
local nRewards = 0

-- start a new game, here screen == state
-- local screen, reward, terminal = gameEnv:getState()
local reward, screen, terminal = gameEnv:step()

-- get model:
local model, criterion
local net = nn.Sequential()
-- layer 1
net:add(nn.SpatialConvolution(1,8,5,5,2,2))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
-- layer 2
net:add(nn.SpatialConvolution(8,16,5,5,1,1))
net:add(nn.ReLU())
-- classifier
net:add(nn.View(16))
-- net:add(nn.Linear(32, 16))
-- net:add(nn.ReLU())
net:add(nn.Linear(16, #gameActions))

-- model, criterion = createModel(#gameActions, opt.sFrames)
model = net
criterion = nn.MSECriterion() 

-- test:
print(model:forward(torch.Tensor(1,24,24)))

print('This is the model:', model)
w, dE_dw = model:getParameters()
print('Number of parameters ' .. w:nElement())
print('Number of grads ' .. dE_dw:nElement())

-- use GPU, if desired:
if opt.useGPU then
  require 'cunn'
  require 'cutorch'
  cutorch.setDevice(opt.gpuId)
  model:cuda()
  criterion:cuda()
  print('Using GPU number', opt.gpuId)
end




--[[ Runs one gradient update using SGD returning the loss.]] --
local function trainNetwork(model, inputs, targets, criterion, sgdParams)
    local loss = 0
    local x, gradParameters = model:getParameters()
    local function feval(x_new)
        gradParameters:zero()
        local predictions = model:forward(inputs)
        local loss = criterion:forward(predictions, targets)
        model:zeroGradParameters()
        local gradOutput = criterion:backward(predictions, targets)
        model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    local _, fs = optim.sgd(feval, x, sgdParams)
    loss = loss + fs[1]
    return loss
end

local epsilon = 1 -- The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)
local epsilonMinimumValue = 0.03 -- The minimum value we want epsilon to reach in training. (0 to 1)
-- local nbActions = 3 -- The number of actions. Since we only have left/stay/right that means 3 actions.
local epochs = 10000 -- The number of games we want the system to run for.
local hiddenSize = 100 -- Number of neurons in the hidden layers.
local maxMemory = 5000 -- How large should the memory be (where it stores its past experiences).
local batchSize = 32 -- The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.
local gridSize = 24 -- The size of the grid that the agent is going to play the game on.
-- local nbStates = gridSize * gridSize -- We eventually flatten to a 1d tensor for the network.
local discount = 0.9 -- The discount is used to force the network to choose states that lead to the reward quicker (0 to 1)


-- Params for Stochastic Gradient Descent (our optimizer).
local sgdParams = {
    learningRate = 0.01,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

-- Mean Squared Error for our loss function.
local criterion = nn.MSECriterion()

-- local env = --CatchEnvironment(gridSize)
local memory = Memory(maxMemory, discount)

local win
local winCount = 0
for i = 1, epochs do
  sys.tic()
  -- Initialize the environment
  local err = 0
  local isGameOver = false

  -- The initial state of the environment
  local currentState = gameEnv:start()

  while (isGameOver ~= true) do
      local action
      -- Decides if we should choose a random action, or an action from the policy network.
      if math.random() < epsilon then
          action = math.random(1, #gameActions)
      else
          -- Forward the current state through the network.
          local q = model:forward(currentState)
          -- Find the max index (the chosen action).
          local max, index = torch.max(q, 1)
          action = index[1]
      end
      -- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
      if (epsilon > epsilonMinimumValue) then
          epsilon = epsilon * (1- 1/(batchSize * epochs))
      end
      local reward, nextState, gameOver = gameEnv:step(action)
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

      -- We get a batch of training data to train the model
      local inputs, targets = memory.getBatch(model, batchSize, #gameActions, gridSize)

      -- Train the network which returns the error
      err = err + trainNetwork(model, inputs, targets, criterion, sgdParams)

      -- display:
      -- win = image.display({image=currentState, zoom=10, win=win})
  end
  if i%10 == 0 then
    print(string.format("Epoch: %d, err: %f, epsilon: %f, Win count: %d, time %f", 
                          i, err, epsilon, winCount, 1000*sys.toc()))
  end
end

torch.save("catch-model.net", model)
