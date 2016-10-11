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
  --actrep              (default 1)                 how many times to repeat action
  --random_starts       (default 0)                 play action 0 between 1 and random_starts number of times at the start of each training episode
<<<<<<< HEAD
  --seed                (default 1250)              initial random seed
  
=======

  Command line options:
  --savedir         (default './results')  subdirectory to save experiments in
  --seed                (default 1250)     initial random seed
  --useGPU                                 use GPU in training
  Data parameters:
  --dataBig                                use large dataset or reduced one

>>>>>>> origin/master
  Training parameters:
  -r,--learningRate       (default 0.001)     learning rate
  -d,--learningRateDecay  (default 0)         learning rate decay
  -w,--weightDecay        (default 0)         L2 penalty on the weights
  -m,--momentum           (default 0.9)       momentum parameter
  --steps                 (default 1e5)       number of training steps to perform
  --epsiUpdate            (default 1e5)       epsilon update
  --prog_freq             (default 1e2)       frequency of progress output
  --save_freq             (default 5e4)       the model is saved every save_freq steps

  Model parameters:
  --lstmLayers            (default 1)     number of layers of RNN / LSTM
  --nSeq                  (default 19)    input video sequence lenght

  Display and save parameters:
  --zoom                  (default 1)     zoom window
  -v, --verbose           (default 2)     verbose output
  --display                               display stuff
  --savedir      (default './results')    subdirectory to save experiments in
]]


torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)
-- print(game_env, #game_actions, agent, opt)

-- set parameters and vars:
local step = 0
local epsilon = 1 -- ϵ-greedy action selection
local gamma = 0.9 -- delayed reward
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


-- online training: algorithm from: http://outlace.com/Reinforcement-Learning-Part-3/
print("Started training...")
local win = nil
while step < opt.steps do
    step = step + 1

    -- learning function for neural net:
    local eval_E = function(w)
      local f = 0
      model:zeroGradParameters()
      f = f + criterion:forward(output, target)
      local dE_dy = criterion:backward(output, target)
      model:backward(screen[1],dE_dy)
      dE_dw:add(opt.weightDecay, w)
      return f, dE_dw -- return f and df/dX
    end

    -- We are in state S
    -- use model to get next action: Q function on S to get Q values for all possible actions
    screen_in = image.scale(screen[1], 84, 84, 'bilinear') -- scale image to smaller size
    output = model:forward(screen_in)
    local value, action_index = output:max(1) -- select max output
    -- print(action_index:size())
    action_index = action_index[1] -- max index is next action!

    -- at random chose random action or action from neural net: best action from Q(S,a)
    if math.random() < epsilon then
      action_index = math.random(#game_actions) -- random action
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

    target = output:clone() -- copy previous output as target

    -- observe Q(S',a)
    if not terminal then
      screen_in = image.scale(screen[1], 84, 84, 'bilinear') -- scale image to smaller size
      output = model:forward(screen_in)
      value, action_index = output:max(1)
      update = reward + gamma*value
      target[action_index[1]] = update -- target is previous output updated with reward

      -- then train neural net:
      _,fs = optim.adam(eval_E, w, optimState)
      err = err + fs[1]
    end

    if step % opt.prog_freq == 0 then
      print('==> iteration = ' .. step ..
        ', number rewards ' .. nrewards .. ', total reward ' .. total_reward ..
        -- string.format(', average loss = %.2f', err) ..
        string.format(', epsilon %.2f', epsilon) .. ', lr '..opt.learningRate )
    end
    err = 0 -- reset error

    -- epsilon is updated every once in a while to do less random actions (and more neural net actions)
    if epsilon > 0.1 then epsilon = epsilon - (1/opt.epsiUpdate) end

    -- display screen
    win = image.display({image=screen, win=win, zoom=opt.zoom})

    -- save results if needed:
    if step % opt.save_freq == 0 then
      torch.save(opt.savedir .. '/DQN_' .. step .. ".t7", 
        {model = model, total_reward = total_reward, nrewards = nrewards})
    end

    if step%1000 == 0 then collectgarbage() end
end
print('Finished training!')
