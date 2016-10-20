-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code
-- an implementation of: http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
-- inspired by: http://outlace.com/Reinforcement-Learning-Part-3/
-- or: https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

-- playing CATCH version:

-- if not dqn then
    -- require "initenv"
-- end

local image = require 'image'
local Catch = require 'rlenvs/Catch'
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
  --batchSize             (default 128)       batch size for training
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
opt.pool_frms = 'type=' .. opt.pool_frms_type .. ',size=' .. opt.pool_frms_size
opt.saveFreq = opt.steps / 10 -- save 10 times in total

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

-- Detect QT for image display
local qt = pcall(require, 'qt')

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
net:add(nn.SpatialConvolution(opt.sFrames,32,3,3,1,1))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
-- layer 2
net:add(nn.SpatialConvolution(32,64,3,3,1,1))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
-- layer 3
net:add(nn.SpatialConvolution(64,64,3,3,1,1))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
-- classifier
net:add(nn.View(64))
net:add(nn.Linear(64, 32))
net:add(nn.ReLU())
net:add(nn.Linear(32, #gameActions))

-- model, criterion = createModel(#gameActions, opt.sFrames)
model = net
criterion = nn.MSECriterion() 

-- test:
-- print(model:forward(torch.Tensor(4,24,24)))

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

--- set up random number generators
-- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
-- RNG seed to the first uniform random int32 from the previous RNG;
-- this is preferred because using the same seed for both generators
-- may introduce correlations; we assume that both torch RNGs ensure
-- adequate dispersion for different seeds.
-- math.random = nil
-- opt.seed = opt.seed or 1
-- torch.manualSeed(opt.seed)
-- if opt.verbose >= 1 then
--     print('Torch Seed:', torch.initialSeed())
-- end
-- local firstRandInt = torch.random()
-- if opt.useGPU then
--     cutorch.manualSeed(firstRandInt)
--     if opt.verbose >= 1 then
--         print('CUTorch Seed:', cutorch.initialSeed())
--     end
-- end


-- online training:
local win = nil
local aHist = torch.zeros(#gameActions)
local input, newinput, output, target, state, newState, outNet, value, actionIdx
local step = 0
local bufStep = 1 -- easy way to keep buffer index
local buffer = {} -- Experience Replay buffer
state = torch.zeros(opt.sFrames, opt.imSize, opt.imSize)
newState = torch.zeros(opt.sFrames, opt.imSize, opt.imSize)
input = torch.zeros(opt.batchSize, opt.sFrames, opt.imSize, opt.imSize)
if opt.useGPU then input = input:cuda() end
newinput = torch.zeros(opt.batchSize, opt.sFrames, opt.imSize, opt.imSize)
if opt.useGPU then newinput = newinput:cuda() end
target = torch.zeros(opt.batchSize, #gameActions)
if opt.useGPU then target = target:cuda() end


-- local logger = optim.Logger('gradient.log')
-- logger:setNames{'dE_dy1', 'dE_dy2', 'dE_dy3', 'dE_dy4'}
-- logger:style{'-', '-', '-', '-'}


print("Started training...")
while step < opt.steps do
  step = step + 1
  sys.tic()

  -- learning function for training our neural net:
  local eval_E = function(w)
    dE_dw:zero()
    local f = criterion:forward(output, target)
    local dE_dy = criterion:backward(output, target)
    -- print(dE_dy[1]:view(1,-1), target)
    -- logger:add(torch.totable(dE_dy)[1])
    -- logger:plot()
    model:backward(input, dE_dy)
    return f, dE_dw -- return f and df/dX
  end

  -- we compute new actions only every few frames
  if step == 1 or step % opt.sFrames == 0 then
    -- We are in state S, now use model to get next action:
    -- game screen size = {1,24,24}
    state[(step/opt.sFrames)%opt.sFrames+1] = screen -- scale screen, average color planes
    if opt.useGPU then state = state:cuda() end
    outNet = model:forward(state)

    -- at random chose random action or action from neural net: best action from Q(state,a)
    if torch.uniform() < epsilon then
      actionIdx = torch.random(#gameActions) -- random action
    else
      value, actionIdx = outNet:max(1) -- select max output
      actionIdx = actionIdx[1] -- select action from neural net
      aHist[actionIdx] = aHist[actionIdx]+1
    end
  end

  -- repeat the move >>> every step <<< (while learning happens only every opt.QLearnFreq)
  if not terminal then
    reward, screen, terminal = gameEnv:step(gameActions[actionIdx])
  else
    screen = gameEnv:start()
    terminal = false
  end

  -- count rewards:
  if reward ~= 0 then
    nRewards = nRewards + 1
    totalReward = totalReward + reward
  end

  -- compute action in newState and save to Experience Replay memory:
  if step > 1 and step % opt.sFrames == 0 then
    -- game screen size = {1,24,24}
    newState[(step/opt.sFrames)%opt.sFrames+1] = screen -- scale screen, average color planes
    if opt.useGPU then state = state:cuda() end
    if opt.useGPU then newState = newState:cuda() end

    -- Experience Replay: store episode in rolling buffer memory (system memory, not GPU mem!)
    buffer[bufStep%opt.ERBufSize] = { state=state:clone():float(), action=actionIdx, reward=reward, 
                                      newState=newState:clone():float(), terminal=terminal }
    -- note 1: this rolling buffer places something in [0] which will not be used later... something to fix at some point...
    -- note 2: find a better way to store episode: store only important episode
    bufStep = bufStep + 1
  end

  -- Q-learning in batch mode every few steps:
  if bufStep > opt.batchSize then -- we shoudl not start training until we have filled the buffer
    -- create next training batch:
    local ri = torch.randperm(#buffer)
    for i = 1, opt.batchSize do
      -- print('indices:', i, ri[i])
      input[i] = opt.useGPU and buffer[ri[i]].state:cuda() or buffer[ri[i]].state
      newinput[i] = opt.useGPU and buffer[ri[i]].newState:cuda() or buffer[ri[i]].newState
    end
    newOutput = model:forward(newinput):clone() -- get output at 'newState' (clone to avoid losing it next model forward!)
    output = model:forward(input) -- get output at state for backprop
    -- print(output)
    -- here we modify the target vector with Q updates:
    local val, update
    for i=1,opt.batchSize do
      target[i] = output[i] -- get target vector at 'state'
      -- print('target:', target[i]:view(1,-1))
      -- update from newState:
      if buffer[ri[i]].terminal then
        update = buffer[ri[i]].reward
      else
        val = newOutput[i]:max() -- computed at 'newState'
        update = buffer[ri[i]].reward + gamma * val
      end
      target[i][buffer[ri[i]].action] = update -- target is previous output updated with reward
      -- print('new target:', target[i]:view(1,-1), 'update', target[i][buffer[ri[i]].action])
      -- print('action', buffer[ri[i]].action, '\n\n\n')
    end
    if opt.useGPU then target = target:cuda() end

    -- then train neural net:
    _,fs = optim.adam(eval_E, w, optimState)
    err = err + fs[1]
  end

  -- epsilon is updated every once in a while to do less random actions (and more neural net actions)
  if epsilon > 0.1 then epsilon = epsilon - (1/opt.steps) end

  -- display screen and print results:
  if opt.display then win = image.display({image=screen, win=win, zoom=opt.zoom, title='Train'}) end
  if step % opt.progFreq == 0 then
    print('==> iteration = ' .. step ..
      ', number rewards ' .. nRewards .. ', total reward ' .. totalReward ..
      string.format(', average loss = %f', err) ..
      string.format(', epsilon %.2f', epsilon) .. ', lr '..opt.learningRate .. 
      string.format(', step time %.2f [ms]', sys.toc()*1000)
    )
    print('Action histogram:', aHist:view(1,#gameActions))
    aHist:zero()
    err = 0 -- reset after reporting period
  end
  

  -- save results if needed:
  if step % opt.saveFreq == 0 then
    torch.save( opt.savedir .. '/DQN_model' .. step .. ".net", model:clone():clearState():float() )
  end


    -- test phase:
  if step % opt.testFreq == 0 and step > 1 then
    local screen, reward, terminal = gameEnv:newGame()

    local testReward = 0
    local nTestRewards = 0
    local nEpisodes = 0

    local testTime = sys.clock()
    for estep = 1, opt.evalSteps do

      local state = screen
      if opt.useGPU then state = state:cuda() end
      local outTest = model:forward(state)

      -- at random chose random action or action from neural net: best action from Q(state,a)
      local v, testIdx = outTest:max(1) -- select max output
      testIdx = testIdx[1] -- select action from neural net
      
      -- Play game in test mode (episodes don't end when losing a life)
      screen, reward, terminal = gameEnv:step(gameActions[testIdx])

      -- display screen
      win2 = image.display({image=screen, win=win2, title='Test'})

      -- record every reward
      testReward = testReward + reward
      if reward ~= 0 then
         nTestRewards = nTestRewards + 1
      end

      if terminal then
          nEpisodes = nEpisodes + 1
          screen, reward, terminal = gameEnv:nextRandomGame()
      end
    end
    testTime = sys.clock() - testTime
    print('Testing iterations = ' .. opt.evalSteps ..
      ', number rewards ' .. testReward .. ', total reward ' .. nTestRewards ..
      string.format(', Testing time %.2f [ms]', testTime*1000)
    )
  end  


  if step%1000 == 0 then collectgarbage() end
end
print('Finished training!')
