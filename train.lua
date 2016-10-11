-- Eugenio Culurciello
-- October 2016

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

  Command line options:
  --savedir         (default './results')  subdirectory to save experiments in
  --seed                (default 1250)     initial random seed
  --useGPU                                 use GPU in training
  Data parameters:
  --dataBig                                use large dataset or reduced one
  
  Training parameters:
  -r,--learningRate       (default 0.001)  learning rate
  -d,--learningRateDecay  (default 0)      learning rate decay
  -w,--weightDecay        (default 0)      L2 penalty on the weights
  -m,--momentum           (default 0.9)    momentum parameter
  --steps               (default 10e5)      number of training steps to perform')

  Model parameters:
  --lstmLayers            (default 1)     number of layers of RNN / LSTM
  --nSeq                  (default 19)    input video sequence lenght
  
  Display and save parameters:
  --zoom                  (default 4)     zoom window
  -v, --verbose           (default 2)     verbose output
  --display                               display stuff
  -s,--save                               save models
  --savePics                              save output images examples
]]


torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)
-- print(game_env, #game_actions, agent, opt)

-- set parameters and vars:
local step = 0


-- start a new game
local screen, reward, terminal = game_env:getState()

-- get model:
net = createModel(#game_actions)
print('This is the model:', net)

-- training:
print("Started training...")
local win = nil
while step < opt.steps do
    step = step + 1
    -- print(screen:size())

    -- use model to get next action:
    local netOut = net:forward(screen)
    local value, action_index = netOut:max(1) -- select max output
    -- print(action_index:size())
    action_index = action_index[1] -- max index is next action!
    -- print(action_index)

    -- game over? get next game!
    if not terminal then
        screen, reward, terminal = game_env:step(game_actions[action_index], true)
    else
        if opt.random_starts > 0 then
            screen, reward, terminal = game_env:nextRandomGame()
        else
            screen, reward, terminal = game_env:newGame()
        end
    end
    
    -- display screen
    win = image.display({image=screen, win=win, zoom=opt.zoom})

    if step%1000 == 0 then collectgarbage() end



end
print('Finished training!')


