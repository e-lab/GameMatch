-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code
-- an implementation of: http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
-- inspired by: http://outlace.com/Reinforcement-Learning-Part-3/

if not dqn then
    require "initenv"
end
require 'pl'
lapp = require 'pl.lapp'
opt = lapp [[

  Game options:
  --framework           (default 'alewrap')         name of training framework
  --env                 (default 'breakout')        name of environment to use')
  --game_path           (default 'roms/')           path to environment file (ROM)
  --env_params          (default 'useRGB=true')     string of environment parameters
  --pool_frms_type      (default 'max')             pool inputs frames mode
  --pool_frms_size      (default '4')               pool inputs frames size
  --actrep              (default 1)                 how many times to repeat action
  --randomStarts        (default 30)                play action 0 between 1 and random_starts number of times at the start of each training episode
  --gamma               (default 0.975)             discount factor in learning
  --epsilon             (default 1)                 initial value of ϵ-greedy action selection
  
  Training parameters:
  --threads               (default 8)         number of threads used by BLAS routines
  --seed                  (default 1)         initial random seed
  -r,--learningRate       (default 0.001)     learning rate
  -d,--learningRateDecay  (default 0)         learning rate decay
  -w,--weightDecay        (default 0)         L2 penalty on the weights
  -m,--momentum           (default 0.9)       momentum parameter
  --batchSize             (default 256)       batch size for training
  --ERBufSize             (default 1e5)       Experience Replay buffer memory
  --QLearnFreq            (default 4)         learn every update_freq steps of game
  --steps                 (default 1e6)       number of training steps to perform
  --progFreq              (default 1e3)       frequency of progress output
  --testFreq              (default 1e3)       frequency of testing
  --evalSteps             (default 1e4)       number of test games to play to test results
  --useGPU                                    use GPU in training

  Display and save parameters:
  --zoom                  (default 4)     zoom window
  -v, --verbose           (default 2)     verbose output
  --display                               display stuff
  --savedir      (default './results')    subdirectory to save experiments in
]]

-- format options:
opt.pool_frms = 'type=' .. opt.pool_frms_type .. ',size=' .. opt.pool_frms_size
opt.epsiFreq = opt.steps -- update epsilon with steps
opt.saveFreq = opt.steps / 10 -- save 10 times in total

if opt.verbose >= 1 then
    print('Using options:')
    for k, v in pairs(opt) do
        print(k, v)
    end
end

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)


--- General setup:
local gameEnv, gameActions, agent, opt = setup(opt)

-- set parameters and vars:
local epsilon = opt.epsilon -- ϵ-greedy action selection
local gamma = opt.gamma -- discount factor
local err = 0 -- loss function error
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
local screen, reward, terminal = gameEnv:getState()

-- get model:
local model, criterion
model, criterion = createModel(#gameActions)
print('This is the model:', model)
w, dE_dw = model:getParameters()
print('Number of parameters ' .. w:nElement())
print('Number of grads ' .. dE_dw:nElement())

-- use GPU, if desired:
if opt.useGPU then
  require 'cunn'
  require 'cutorch'
  model:cuda()
  criterion:cuda()
  print('Using GPU')
end

--- set up random number generators
-- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
-- RNG seed to the first uniform random int32 from the previous RNG;
-- this is preferred because using the same seed for both generators
-- may introduce correlations; we assume that both torch RNGs ensure
-- adequate dispersion for different seeds.
math.random = nil
opt.seed = opt.seed or 1
torch.manualSeed(opt.seed)
if opt.verbose >= 1 then
    print('Torch Seed:', torch.initialSeed())
end
local firstRandInt = torch.random()
if opt.useGPU then
    cutorch.manualSeed(firstRandInt)
    if opt.verbose >= 1 then
        print('CUTorch Seed:', cutorch.initialSeed())
    end
end


-- online training: algorithm from: http://outlace.com/Reinforcement-Learning-Part-3/
local win = nil
local input, newinput, output, target, state, outNet, value, actionIdx
local step = 0
local bufStep = 1 -- easy way to keep buffer index
local buffer = {} -- Experience Replay buffer
input = torch.Tensor(opt.batchSize, 3, 84, 84)
if opt.useGPU then input = input:cuda() end
newinput = torch.Tensor(opt.batchSize, 3, 84, 84)
if opt.useGPU then newinput = newinput:cuda() end
target = torch.Tensor(opt.batchSize, #gameActions)
if opt.useGPU then target = target:cuda() end

print("Started training...")
while step < opt.steps do
  step = step + 1
  sys.tic()

  -- learning function for training our neural net:
  local eval_E = function(w)
    local f = 0
    model:zeroGradParameters()
    f = f + criterion:forward(output, target)
    local dE_dy = criterion:backward(output, target)
    model:backward(input,dE_dy)
    return f, dE_dw -- return f and df/dX
  end

  -- we compute new actions only every few frames
  if step == 1 or step % opt.QLearnFreq == 0 then
    -- We are in state S, now use model to get next action:
    -- game screen size = {1,3,210,160}
    state = image.scale(screen[1], 84, 84) -- scale screen
    -- state = image.scale(screen[1][{{},{94,194},{9,152}}], 84, 84) -- scale screen -- resize to smaller portion
    -- win = image.display({image=state, win=win, zoom=opt.zoom}) -- debug line
    if opt.useGPU then state = state:cuda() end
    outNet = model:forward(state)

    -- at random chose random action or action from neural net: best action from Q(state,a)
    if torch.uniform() < epsilon then
      actionIdx = torch.random(#gameActions) -- random action
    else
      value, actionIdx = outNet:max(1) -- select max output
      actionIdx = actionIdx[1] -- select action from neural net
    end
  end

  -- repeat the move >>> every step <<< (while learning happens only every opt.QLearnFreq)
  if not terminal then
      screen, reward, terminal = gameEnv:step(gameActions[actionIdx], true)
  else
      if opt.randomStarts > 0 then
          screen, reward, terminal = gameEnv:nextRandomGame()
      else
          screen, reward, terminal = gameEnv:newGame()
      end
  end

  -- compute action in newState and save to Experience Replay memory:
  if step > 1 and step % opt.QLearnFreq == 0 then
    -- game screen size = {1,3,210,160}
    local newState = image.scale(screen[1], 84, 84) -- scale screen
    -- local newState = image.scale(screen[1][{{},{94,194},{9,152}}], 84, 84) -- scale screen -- resize to smaller portion
    if opt.useGPU then newState = newState:cuda() end
    if reward ~= 0 then
      nRewards = nRewards + 1
      totalReward = totalReward + reward
    end

    -- Experience Replay: store episode in rolling buffer memory (system memory, not GPU mem!)
    buffer[bufStep%opt.ERBufSize] = { state=state:clone():float(), action=actionIdx, outState = outNet:clone():float(),
              reward=reward, newState=newState:clone():float(), terminal=terminal }
    -- note 1: this rolling buffer places something in [0] which will not be used later... something to fix at some point...
    -- note 2: find a better way to store episode: store only important episode
    bufStep = bufStep + 1
  end

  -- Q-learning in batch mode every few steps:
  if step % opt.QLearnFreq == 0 and bufStep > opt.batchSize then -- we shoudl not start training until we have filled the buffer
    -- create next training batch:
    -- print(#buffer)
    local ri = torch.randperm(#buffer)
    for i=1,opt.batchSize do
      -- print('indices:', i, ri[i])
      input[i] = buffer[ri[i]].state:cuda()
      newinput[i] = buffer[ri[i]].newState:cuda()
    end
    -- get output at 'newState'
    output = model:forward(newinput)
    -- here we modify the target vector with Q updates:
    local val, update
    for i=1,opt.batchSize do
      target[i] = buffer[ri[i]].outState -- get target vector at 'state'
      -- observe Q(newState,a)
      if not buffer[ri[i]].terminal then
        val = output[i]:max() -- computed at 'newState'
        update = buffer[ri[i]].reward + gamma * val
      else
        update = buffer[ri[i]].reward
      end
      target[i][buffer[ri[i]].action] = update -- target is previous output updated with reward
    end
    if opt.useGPU then target = target:cuda() end

    -- then train neural net:
    _,fs = optim.adam(eval_E, w, optimState)
    err = err + fs[1]
  end

  -- epsilon is updated every once in a while to do less random actions (and more neural net actions)
  if epsilon > 0.1 then epsilon = epsilon - (1/opt.epsiFreq) end

  -- display screen and print results:
  if opt.display then win = image.display({image=screen, win=win, zoom=opt.zoom, title='Train'}) end
  if step % opt.progFreq == 0 then
    print('==> iteration = ' .. step ..
      ', number rewards ' .. nRewards .. ', total reward ' .. totalReward ..
      -- string.format(', average loss = %.2f', err) ..
      string.format(', epsilon %.2f', epsilon) .. ', lr '..opt.learningRate .. 
      string.format(', error %f', err) ..
      string.format(', step time %.2f [ms]', sys.toc()*1000)
    )
  end
  err = 0 -- reset error

  -- save results if needed:
  if step % opt.saveFreq == 0 then
    torch.save(opt.savedir .. '/DQN_' .. step .. ".t7", 
      {model = model, totalReward = totalReward, nRewards = nRewards})
  end


    -- test phase:
  if step % opt.testFreq == 0 and step > 1 then
    local screen, reward, terminal = gameEnv:newGame()

    local testReward = 0
    local nTestRewards = 0
    local nEpisodes = 0

    local testTime = sys.clock()
    for estep = 1, opt.evalSteps do

      local state = image.scale(screen[1], 84, 84) -- scale screen
      -- state = image.scale(screen[1][{{},{94,194},{9,152}}], 84, 84) -- scale screen -- resize to smaller portion
      if opt.useGPU then state = state:cuda() end
      local outTest = model:forward(state)

      -- at random chose random action or action from neural net: best action from Q(state,a)
      local v, testIdx = outTest:max(1) -- select max output
      testIdx = testIdx[1] -- select action from neural net
      
      -- Play game in test mode (episodes don't end when losing a life)
      screen, reward, terminal = gameEnv:step(gameActions[4])--testIdx])

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
