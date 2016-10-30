-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code

-- playing CATCH version:
-- https://github.com/Kaixhin/rlenvs

-- CNN version


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
  --gamma               (default 0.9)         discount factor in learning
  --epsilon             (default 1)           initial value of ϵ-greedy action selection
  
  Training parameters:
  --threads               (default 8)         number of threads used by BLAS routines
  --seed                  (default 1)         initial random seed
  -r,--learningRate       (default 0.1)       learning rate
  -d,--learningRateDecay  (default 1e-9)      learning rate decay
  -w,--weightDecay        (default 0)         L2 penalty on the weights
  -m,--momentum           (default 0.9)       momentum parameter
  --gridSize              (default 10)        state is screen resized to this size 
  --batchSize             (default 32)        batch size for training
  --maxMemory             (default 1e4)       Experience Replay buffer memory
  --sFrames               (default 4)         input frames to stack as input / learn every update_freq steps of game
  --epochs                (default 1e5)       number of training steps to perform
  --progFreq              (default 1e2)       frequency of progress output
  --useGPU                                    use GPU in training
  --gpuId                 (default 1)         which GPU to use
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
os.execute('mkdir '..opt.savedir)

-- Clamps a number to within a certain range.
function math.clamp(n, low, high) return math.min(math.max(low, n), high) end


--[[ The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
--   and creates batches of experiences based on the mini-batch size for training.]] --
local function Memory(maxMemory)
    local memory = {}

    -- Appends the experience to the memory.
    function memory.remember(memoryInput)
        table.insert(memory, memoryInput)
        if (#memory > opt.maxMemory) then
            -- Remove the earliest memory to allocate new experience to memory.
            table.remove(memory, 1)
        end
    end

    function memory.getBatch(model, gamma, batchSize, nbActions, gridSize)

        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch).
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)

        local inputs = torch.zeros(chosenBatchSize, opt.sFrames, gridSize, gridSize)
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
                target[memoryInput.action] = memoryInput.reward + gamma * nextStateMaxQ
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
local gameEnv = Catch({size = opt.gridSize, level = 1})
local stateSpec = gameEnv:getStateSpec()
local actionSpec = gameEnv:getActionSpec()
local observation = gameEnv:start()
print('screen size is:', observation:size())
-- print({stateSpec}, {actionSpec})
local gameActions = {0,1,2} -- game actions from CATCH
-- print(gameActions, #gameActions)

-- set parameters and vars:
local epsilon = opt.epsilon -- ϵ-greedy action selection
local gamma = opt.gamma -- discount factor
local totalReward = 0
local nRewards = 0


-- get model:
local model
  model = nn.Sequential()
  model:add(nn.SpatialConvolution(opt.sFrames, 32, 4,4, 2,2))
  model:add(nn.SpatialMaxPooling(2,2, 2,2))
  model:add(nn.View(32*4))
  model:add(nn.Linear(32*4, 256))
  model:add(nn.ReLU())
  model:add(nn.Linear(256, #gameActions))
local criterion = nn.MSECriterion() 
-- test:
print(model:forward(torch.Tensor(opt.sFrames,opt.gridSize,opt.gridSize)))
print('This is the model:', model)


-- use GPU, if desired:
if opt.useGPU then
  require 'cunn'
  require 'cutorch'
  cutorch.setDevice(opt.gpuId)
  model:cuda()
  criterion:cuda()
  print('Using GPU number', opt.gpuId)
end


-- training function:
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

-- Params for Stochastic Gradient Descent (our optimizer).
local sgdParams = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}


local memory = Memory(opt.maxMemory)
local epsilon = opt.epsilon
local epsilonMinimumValue = 0.005
local win
local winCount = 0
local totalCount = 0
local currentState = torch.zeros(opt.sFrames, opt.gridSize, opt.gridSize)
local nextState = torch.zeros(opt.sFrames, opt.gridSize, opt.gridSize)
local screen, action, reward, gameOver

for game = 1, opt.epochs do
  sys.tic()
  -- Initialize the environment
  local err = 0
  local isGameOver = false

  -- The initial state of the environment
  screen = gameEnv:start()
  -- currentState = getSimpleState(screen)
  currentState[opt.sFrames] = screen

  while (isGameOver ~= true) do
      -- random action or an action from the policy network:
      if torch.random() < epsilon then
          action = torch.random(#gameActions)
      else
          -- Forward the current state through the network:
          local q = model:forward(currentState)
          -- Find the max index (the chosen action):
          local max, index = torch.max(q, 1)
          action = index[1]
      end

      reward, screen, gameOver = gameEnv:step(gameActions[action])
      -- nextState = getSimpleState(screen)
      for i=1,opt.sFrames-1 do nextState[i] = nextState[i+1] end -- prepare last opt.sFrames frames in sequence
      nextState[opt.sFrames] = screen

      -- count rewards:
      if (reward == 1) then winCount = winCount + 1 end
      -- add current play to experience replay memory
      memory.remember({
          inputState = currentState:clone(),
          action = action,
          reward = reward,
          nextState = nextState:clone(),
          gameOver = gameOver
      })
      -- Update the current state and if the game is over:
      currentState = nextState
      isGameOver = gameOver

      -- get a batch of training data to train the model:
      local inputs, targets = memory.getBatch(model, opt.gamma, opt.batchSize, #gameActions, opt.gridSize)

      -- Train the network, get error: (only train after replay emmeory has been filled)
      if game > opt.maxMemory then
        err = err + trainNetwork(model, inputs, targets, criterion, sgdParams)
      end

      -- display:
      if opt.display then win = image.display({image=screen, zoom=10, win=win, title='Train'}) end
  end
  if epsilon > epsilonMinimumValue then epsilon = epsilon - (opt.epsilon-epsilonMinimumValue)/opt.epochs end -- epsilon update
  if game%opt.progFreq==0 then 
    totalCount = totalCount + winCount
    print(string.format("Epoch: %d, err: %.3f, epsilon: %.2f, Accuracy: %.2f, Win count: %d, Total win count: %d, time %.3f", game, err, epsilon, winCount/opt.progFreq, winCount, totalCount, sys.toc()))
    winCount = 0
  end
end

torch.save("catch-model-dead-simple.net", model)
