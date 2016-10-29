-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code
-- an implementation of: http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
-- inspired by: http://outlace.com/Reinforcement-Learning-Part-3/
-- or: https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

-- playing CATCH version:
-- https://github.com/Kaixhin/rlenvs


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
  --gridSize              (default 24)        state is screen resized to this size 
  --batchSize             (default 32)        batch size for training
  --ERBufSize             (default 1e3)       Experience Replay buffer memory
  --sFrames               (default 4)         input frames to stack as input / learn every update_freq steps of game
  --epochs                (default 1e4)       number of training games to play
  --progFreq              (default 1e2)       frequency of progress output
  --useGPU                                    use GPU in training
  --gpuId                 (default 1)         which GPU to use
  --largeSimple                               simple model or not

  Display and save parameters:
  --zoom                  (default 10)    zoom window
  -v, --verbose           (default 2)     verbose output
  --display                               display stuff
  --savedir      (default './results')    subdirectory to save experiments in
]]

-- format options:
opt.pool_frms = 'type=' .. opt.pool_frms_type .. ',size=' .. opt.pool_frms_size
opt.saveFreq = opt.epochs / 10 -- save 10 times in total

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)

-- Clamps a number to within a certain range.
function math.clamp(n, low, high) return math.min(math.max(low, n), high) end

--- General setup:
-- local gameEnv, gameActions, agent, opt = setup(opt)
local gameEnv = Catch({size = opt.gridSize, level = 2})
local stateSpec = gameEnv:getStateSpec()
local actionSpec = gameEnv:getActionSpec()
local observation = gameEnv:start()
print('screen size is:', observation:size())
-- print(stateSpec,actionSpec)
local gameActions = {0,1,2} -- game actions from CATCH
-- print(gameActions, #gameActions)

-- start a new game, here screen == state
local reward, screen, terminal = gameEnv:step()

-- get model:
local model
if opt.largeModel then
  model = nn.Sequential()
  -- layer 1
  model:add(nn.SpatialConvolution(opt.sFrames,32,3,3,1,1))
  model:add(nn.ReLU())
  mdel:add(nn.SpatialMaxPooling(2,2,2,2))
  -- layer 2
  model:add(nn.SpatialConvolution(32,64,3,3,1,1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  -- layer 3
  model:add(nn.SpatialConvolution(64,64,3,3,1,1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  -- classifier
  model:add(nn.View(64))
  model:add(nn.Linear(64, 32))
  model:add(nn.ReLU())
  model:add(nn.Linear(32, #gameActions))
else
  model = nn.Sequential()
  -- layer 1
  model:add(nn.SpatialConvolution(opt.sFrames,8,5,5,2,2))
  -- model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  -- layer 2
  model:add(nn.SpatialConvolution(8,16,5,5,1,1))
  -- model:add(nn.ReLU())
  -- classifier
  model:add(nn.View(16))
  model:add(nn.Linear(16, #gameActions))
end
local criterion = nn.MSECriterion() 

-- test:
-- print('Test model is:', model:forward(torch.Tensor(4,24,24)))
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

-- set parameters and vars:
local epsilon = opt.epsilon -- ϵ-greedy action selection

local optimState = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
  weightDecay = opt.weightDecay
}
local totalReward = 0
local nRewards = 0

-- online training:
local win = nil
local aHist = torch.zeros(#gameActions)
local ERmemory = {} -- Experience Replay memory
local state = torch.zeros(opt.sFrames, opt.gridSize, opt.gridSize)
local nextState = torch.zeros(opt.sFrames, opt.gridSize, opt.gridSize)
local nextInput = torch.zeros(opt.batchSize, opt.sFrames, opt.gridSize, opt.gridSize)
-- if opt.useGPU then newinput = newinput:cuda() end
local target = torch.zeros(opt.batchSize, #gameActions)
-- if opt.useGPU then target = target:cuda() end



function getBatch(memory, model, batchSize, nbActions, gridSize)
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
    local target = model:forward(memoryInput.state)

    --Gives us Q_sa, the max q for the next state.
    local nextStateMaxQ = torch.max(model:forward(memoryInput.nextState), 1)[1]
    nextStateMaxQ = math.clamp(nextStateMaxQ, -1, 1) -- clamp updates to keep neural net from exploding
    if (memoryInput.gameOver) then
        target[memoryInput.action] = memoryInput.reward
    else
        -- reward + discount(gamma) * max_a' Q(s',a')
        -- We are setting the Q-value for the action to  r + γmax a’ Q(s’, a’). The rest stay the same
        -- to give an error of 0 for those outputs.
        target[memoryInput.action] = memoryInput.reward + opt.gamma * nextStateMaxQ
    end
    -- Update the inputs and targets.
    inputs[i] = memoryInput.state
    targets[i] = target
  end
  return inputs, targets
end


-- training function:
local function trainNetwork(model, inputs, targets, criterion, sgdParams)
    local loss = 0
    local w, dE_dw = model:getParameters()
    local function eval_E(w)
        dE_dw:zero()
        local predictions = model:forward(inputs)
        local loss = criterion:forward(predictions, targets)
        model:zeroGradParameters()
        local gradOutput = criterion:backward(predictions, targets)
        model:backward(inputs, gradOutput)
        return loss, dE_dw
    end

    local _, fs = optim.adam(eval_E, w, optimState)
    loss = loss + fs[1]
    return loss
end


print("Started training...")
-- local w, dE_dw = model:getParameters()
-- local logger = optim.Logger('gradient.log')
-- logger:setNames{'dE_dy1', 'dE_dy2', 'dE_dy3', 'dE_dy4'}
-- logger:style{'-', '-', '-', '-'}
for game = 1, opt.epochs do
  
  sys.tic()
  -- Initialize the environment
  local err = 0
  local GameOver = false

  state[1] = gameEnv:start()

  while not GameOver do
    local action
    -- we compute new actions only every few frames
    -- We are in state S, now use model to get next action:
    -- game screen size = {1,24,24}
    -- if opt.useGPU then state = state:cuda() end
  
    -- at random chose random action or action from neural net: best action from Q(state,a)
    if torch.uniform() < epsilon then
      action = torch.random(#gameActions) -- random action
    else
      local q = model:forward(state)
      local max, index = q:max(1) -- select max output
      action = index[1] -- select action from neural net
      aHist[action] = aHist[action]+1
    end

    -- make the next move:
    reward, screen, terminal = gameEnv:step(gameActions[action])
    for i=1,opt.sFrames-1 do nextState[i] = nextState[i+1] end -- prepare last opt.sFrames frames in sequence
    nextState[opt.sFrames] = screen

    reward = math.clamp(reward, -1, 1) -- clamp reward to keep neural net from exploding

    -- count rewards:
    if reward ~= 0 then
      nRewards = nRewards + 1
      totalReward = totalReward + reward
    end

    -- compute action in newState and save to Experience Replay memory:
    -- game screen size = {1,24,24}
    -- if opt.useGPU then newState = newState:cuda() end

    -- Experience Replay: store episode in rolling buffer memory (system memory, not GPU mem!)
    table.insert( ERmemory, { state=state:clone():float(), action=action, reward=reward, 
                              nextState=nextState:clone():float(), terminal=terminal } )

    -- Update the current state and if the game is over:
    nextState = state
    GameOver = terminal

    -- get a batch of training data to train the model:
    local inputs, targets = getBatch(ERmemory, model, opt.batchSize, #gameActions, opt.gridSize)

    -- then train neural net:
    err = err + trainNetwork(model, inputs, targets, criterion, optimState)
  
    if opt.display then win = image.display({image=screen, win=win, zoom=opt.zoom, title='Train'}) end

  end

  -- epsilon is updated every once in a while to do less random actions (and more neural net actions)
  if epsilon > 0.05 then epsilon = epsilon*(1-1/opt.epochs) end

  -- display screen and print results:
  if game % opt.progFreq == 0 then
    print('==> Game number: ' .. game ..
      ', number rewards ' .. nRewards .. ', total reward: ' .. totalReward ..
      string.format(', average loss: %f', err) ..
      string.format(', epsilon: %.2f', epsilon) .. ', lr: '..opt.learningRate .. 
      string.format(', step time: %.2f [ms]', sys.toc()*1000)
    )
    print('Action histogram:', aHist:view(1,#gameActions))
    aHist:zero()
    nRewards = 0 -- reset this time rewards
    err = 0 -- reset after reporting period
  end
  
  -- save results if needed:
  if game % opt.saveFreq == 0 then
    torch.save( opt.savedir .. '/catch_model' .. game .. ".net", model:clone():clearState():float() )
  end

  if game%1000 == 0 then collectgarbage() end
end
print('Finished training!')
