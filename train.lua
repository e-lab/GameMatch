-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code

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
  --pool_frms_size      (default '2')                 pool inputs frames size
  --actrep              (default 1)                 how many times to repeat action
  --randomStarts        (default 0)                 play action 0 between 1 and random_starts number of times at the start of each training episode
  --gamma               (default 0.975)             discount factor in learning
  --epsilon             (default 1)                 initial value of ϵ-greedy action selection
  
  Training parameters:
  --threads               (default 8)         number of threads used by BLAS routines
  --seed                  (default 1250)      initial random seed
  -r,--learningRate       (default 0.001)     learning rate
  -d,--learningRateDecay  (default 0)         learning rate decay
  -w,--weightDecay        (default 0)         L2 penalty on the weights
  -m,--momentum           (default 0.9)       momentum parameter
  --batchSize             (default 128)       batch size for training
  --ERBufSize             (default 256)       Experience Replay buffer memory
  --QLearnFreq            (default 4)         learn every update_freq steps of game
  --steps                 (default 1e5)       number of training steps to perform
  --epsiFreq              (default 1e5)       epsilon update
  --progFreq              (default 1e2)       frequency of progress output
  --saveFreq              (default 1e4)       the model is saved every save_freq steps
  --useGPU                                    use GPU in training

  Model parameters:
  --lstmLayers            (default 1)     number of layers of RNN / LSTM
  --nSeq                  (default 19)    input video sequence lenght

  Display and save parameters:
  --zoom                  (default 4)     zoom window
  -v, --verbose           (default 2)     verbose output
  --display                               display stuff
  --savedir      (default './results')    subdirectory to save experiments in
]]

-- format options:
opt.pool_frms = 'type=' .. opt.pool_frms_type .. ',size=' .. opt.pool_frms_size

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
local step = 0
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
local total_reward = 0
local nrewards = 0

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

-- online training: algorithm from: http://outlace.com/Reinforcement-Learning-Part-3/
local win = nil
local input, newinput, output, target, value, actionIdx, state, action, newState
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

  -- learning function for neural net:
  local eval_E = function(w)
    local f = 0
    model:zeroGradParameters()
    f = f + criterion:forward(output, target)
    local dE_dy = criterion:backward(output, target)
    model:backward(input,dE_dy)
    return f, dE_dw -- return f and df/dX
  end
      
  -- We are in state S, now use model to get next action:
  state = image.scale(screen[1], 84, 84, 'bilinear') -- scale screen
  if opt.useGPU then state = state:cuda() end
  local outNet = model:forward(state)

  -- at random chose random action or action from neural net: best action from Q(state,a)
  if math.random() < epsilon then
    actionIdx = math.random(#gameActions) -- random action
  else
    value, actionIdx = outNet:max(1) -- select max output
    actionIdx = actionIdx[1] -- select action from neural net
  end

  -- make the move:
  if not terminal then
      screen, reward, terminal = gameEnv:step(gameActions[actionIdx], true)
  else
      if opt.randomStarts > 0 then
          screen, reward, terminal = gameEnv:nextRandomGame()
      else
          screen, reward, terminal = gameEnv:newGame()
      end
  end
  newState = image.scale(screen[1], 84, 84, 'bilinear') -- scale screen
  if opt.useGPU then newState = newState:cuda() end
  if reward ~= 0 then
    nrewards = nrewards + 1
    total_reward = total_reward + reward
  end

  -- Experience Replay: store episode in rolling buffer memory:
  buffer[step%opt.ERBufSize+1] = {state=state:clone(), action=actionIdx, outState = outNet:clone(),
            reward=reward, newState=newState:clone(), terminal=terminal}
  -- note: this rolling buffer places something in [0] which will not be used later... something to fix at some point...

  -- Q-learning in batch mode every few steps:
  if step % opt.QLearnFreq == 0 and step > opt.ERBufSize then -- we shoudl not start training until we have filled the buffer
    -- print('step', step)
    -- print('buffer size', #buffer)
    -- create next training batch:
    local ri = torch.randperm(opt.ERBufSize)
    for i=1,opt.batchSize do
      -- print('indices', i, ri[i])
      input[i] = buffer[ri[i]].state
      newinput[i] = buffer[ri[i]].newState
    end
    -- get output at 'state'
    output = model:forward(newinput)
    -- here we modify the target vector with Q updates:
    for i=1,opt.batchSize do
      target[i] = buffer[ri[i]].outState -- get target vector at 'state'
      -- observe Q(newState,a)
      if not buffer[ri[i]].terminal then
        value, actionIdx = output[i]:max(1) -- computed at 'newState'
        update = buffer[ri[i]].reward + gamma*value
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

  if step % opt.progFreq == 0 then
    print('==> iteration = ' .. step ..
      ', number rewards ' .. nrewards .. ', total reward ' .. total_reward ..
      -- string.format(', average loss = %.2f', err) ..
      string.format(', epsilon %.2f', epsilon) .. ', lr '..opt.learningRate ..
      string.format(', step time %.2f [ms]', sys.toc()*1000) 
    )
  end
  err = 0 -- reset error

  -- epsilon is updated every once in a while to do less random actions (and more neural net actions)
  if epsilon > 0.1 then epsilon = epsilon - (1/opt.epsiFreq) end

  -- display screen
  if opt.display then win = image.display({image=screen, win=win, zoom=opt.zoom}) end

  -- save results if needed:
  if step % opt.saveFreq == 0 then
    torch.save(opt.savedir .. '/DQN_' .. step .. ".t7", 
      {model = model, total_reward = total_reward, nrewards = nrewards})
  end

  if step%1000 == 0 then collectgarbage() end
end
print('Finished training!')
