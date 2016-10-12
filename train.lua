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
  --random_starts       (default 0)                 play action 0 between 1 and random_starts number of times at the start of each training episode
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
  --ERBuffer              (default 1024)      Experience Replay buffer memory
  --update_freq           (default 4)         learn every update_freq steps of game
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
local game_env, game_actions, agent, opt = setup(opt)

-- set parameters and vars:
local step = 0
local epsilon = opt.epsilon -- ϵ-greedy action selection
local gamma = opt.gamma -- discount factor
local err = 0 -- loss function error
local w, dE_dw
local optimState = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay
}
local total_reward = 0
local nrewards = 0

-- start a new game, here screen == state
local screen, reward, terminal = game_env:getState()

-- get model:
local model, criterion
model, criterion = createModel(#game_actions)
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
local input, output, target, value, action_index
local buffer = {} -- Experience Replay buffer

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
    dE_dw:add(opt.weightDecay, w)
    return f, dE_dw -- return f and df/dX
  end

  local function modelEval()
      input = image.scale(screen[1], 84, 84, 'bilinear') -- scale image to smaller size
      if opt.useGPU then input = input:cuda() end
      output = model:forward(input)
  end

  local function QLearn()
    target = output:clone() -- copy previous output as target

    -- observe Q(S',a)
    if not terminal then
      modelEval()
      value, action_index = output:max(1)
      update = reward + gamma*value
      target[action_index[1]] = update -- target is previous output updated with reward

      -- then train neural net:
      _,fs = optim.adam(eval_E, w, optimState)
      err = err + fs[1]
    end
  end
      
  -- We are in state S
  -- use model to get next action: Q function on S to get Q values for all possible actions
  modelEval()

  -- at random chose random action or action from neural net: best action from Q(S,a)
  if math.random() < epsilon then
    action_index = math.random(#game_actions) -- random action
  else
    value, action_index = output:max(1) -- select max output
    action_index = action_index[1] -- select action from neural net
  end

  -- make the move:
  if not terminal then
      screen, reward, terminal = game_env:step(game_actions[action_index], true)
  else
      if opt.random_starts > 0 then
          screen, reward, terminal = game_env:nextRandomGame()
      else
          screen, reward, terminal = game_env:newGame()
      end
  end
  if reward ~= 0 then
    nrewards = nrewards + 1
    total_reward = total_reward + reward
  end

  -- Q-learning updates every few steps:
  if step % opt.update_freq == 0 then
    QLearn()
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
