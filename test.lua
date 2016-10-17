-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code
-- an implementation of: http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
-- inspired by: http://outlace.com/Reinforcement-Learning-Part-3/
-- test trained net

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

  Net options:
  --loadNet             (default '')                trained neural network to load
  --imSize                (default 84)        state is screen resized to this size 
  --sFrames             (default 4)         input frames to stack as input 

  Display and save parameters:
  --zoom                  (default 4)     zoom window
  -v, --verbose           (default 2)     verbose output
  --display                               display stuff
  --savedir      (default './results-test')    subdirectory to save experiments in
]]

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
os.execute('mkdir '..opt.savedir)

--- General setup:
local gameEnv, gameActions, agent, opt = setup(opt)

--load model:
local model = torch.load(opt.loadNet)
print('This is the model:', model)

-- start a new game, here screen == state
local screen, reward, terminal = gameEnv:getState()

-- online training: algorithm from: http://outlace.com/Reinforcement-Learning-Part-3/
local win = nil
local actionIdx = torch.random(#gameActions) -- random action

local step = 0
local state
state = torch.zeros(opt.sFrames, opt.imSize, opt.imSize)

print("Started testing network...")
while true do
  step = step + 1
  sys.tic()

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

    -- we compute new actions only every few frames
  if step == 1 or step % opt.sFrames == 0 then
    -- game screen size = {1,3,210,160}
    state[step%opt.sFrames+1] = image.scale(screen[1], opt.imSize, opt.imSize):sum(1):div(3) -- scale screen, average color planes
    outNet = model:forward(state)
    value, actionIdx = outNet:max(1) -- select max output
    actionIdx = actionIdx[1] -- select action from neural net
  end

  win = image.display({image=screen, win=win, zoom=opt.zoom, title='Test net'})

  if step%1000 == 0 then collectgarbage() end
end
print('Finished testing network!')
