-- Eugenio Culurciello
-- October 2016

require 'qtwidget' -- for keyboard interaction
if not dqn then
    require "initenv"
end
require 'image'
require 'pl'
lapp = require 'pl.lapp'
opt = lapp [[

  Game options:
  --gridSize            (default 20)          game grid size 
  --discount            (default 0.9)         discount factor in learning
  --epsilon             (default 1)           initial value of ϵ-greedy action selection
  --epsilonMinimumValue (default 0.001)       final value of ϵ-greedy action selection
  --playFile            (default '')          human play file to initialize exp. replay memory
  --framework           (default 'alewrap')         name of training framework
  --env                 (default 'breakout')        name of environment to use')
  --game_path           (default 'roms/')           path to environment file (ROM)
  --env_params          (default 'useRGB=true')     string of environment parameters
  --pool_frms_type      (default 'max')             pool inputs frames mode
  --pool_frms_size      (default '1')               pool inputs frames size
  --actrep              (default 4)                 how many times to repeat action, frames to skip to speed up game and inference
  --randomStarts        (default 30)                play action 0 between 1 and random_starts number of times at the start of each training episode
 
  Training parameters:
  --threads               (default 8)         number of threads used by BLAS routines
  --seed                  (default 1)         initial random seed

  Model parameters:
  --fw                                        Use FastWeights or not
  --nLayers               (default 1)         RNN layers
  --nHidden               (default 128)       RNN hidden size
  --nFW                   (default 8)         number of fast weights previous vectors

  Display and save parameters:
  --zoom                  (default 4)        zoom window
  -v, --verbose           (default 2)        verbose output
  --display                                  display stuff
  --savedir          (default './results')   subdirectory to save experiments in
  --progFreq              (default 1e2)       frequency of progress output
]]

local nbStates = opt.gridSize * opt.gridSize
local nSeq = 4*opt.gridSize -- RNN max sequence length in this game is grid size

-- format options:
opt.pool_frms = 'type=' .. opt.pool_frms_type .. ',size=' .. opt.pool_frms_size

if opt.verbose >= 1 then
    print('Using options:')
    for k, v in pairs(opt) do
        print(k, v)
    end
end

package.path = '../catch/?.lua;' .. package.path
local rnn = require 'RNN'

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)

local gameEnv, gameActions, agent, opt = setup(opt) -- setup game environment
print('Game started. Number of game actions:', #gameActions)
local nbActions = #gameActions
local nbStates = opt.gridSize * opt.gridSize
local nSeq = 4*opt.gridSize -- RNN max sequence length in this game is grid size

local qtimer = qt.QTimer()

function preProcess(im)
  local net = nn.SpatialMaxPooling(4,4,4,4)
  -- local pooled = net:forward(im[1])
  local pooled = net:forward(im[1][{{},{94,194},{9,152}}])
  -- print(pooled:size())
  local out = image.scale(pooled, opt.gridSize, opt.gridSize):sum(1):div(3)
  return out
end

-- start a new game
local screen, reward, isGameOver = gameEnv:newGame()
local currentState = preProcess(screen) -- resize to smaller size
local episodes, totalReward = 0, 0

-- Create a window for displaying output frames
win = qtwidget.newwindow(opt.zoom*screen:size(4), opt.zoom*screen:size(3),'EC game engine')

local action = 2
local seqMem = torch.Tensor(nSeq, nbStates) -- store sequence of states in successful run
local seqAct = torch.zeros(nSeq, nbActions) -- store sequence of actions in successful run
local memory = {} -- memory to save play data

local isGameOver = false
local steps = 0 -- count steps to game win

function main()
  if steps >= nSeq then steps = 0 end
  steps = steps+1
  -- look at screen:
  win = image.display({image=screen, zoom=opt.zoom, win=win})

  -- get human player move:
  -- currentState, reward, isGameOver = gameEnv.act(action)
  screen, reward, isGameOver = gameEnv:step(gameActions[action], false)
  local currentState = preProcess(screen) -- resize to smaller size

  -- store to memory
  seqMem[steps] = currentState:clone() -- store state sequence into memory
  seqAct[steps][action] = 1

  action = 2 -- stop move
  
  if reward == 1 then 
    totalReward = totalReward + reward
    print('Total Reward:', totalReward)
    table.insert(memory, {states = seqMem:byte(), actions = seqAct:byte()}) -- insert successful sequence into data memory
  end

  if isGameOver then
    episodes = episodes + 1
    gameEnv.reset()
    isGameOver = false
    steps = 0
  end
end

-- game controls for Breakout
print('Game controls: left / right')
qtimer.interval = 300
qtimer.singleShot = false
qt.connect(qtimer,'timeout()', main)

qt.connect(win.listener,
         'sigKeyPress(QString, QByteArray, QByteArray)',
         function(_, keyValue)
            if keyValue == 'Key_Right' then
                action = 3
            elseif keyValue == 'Key_Left' then
                action = 4
            elseif keyValue == 'Key_X' then
                action = 1
            elseif keyValue == 'Key_Q' then
                torch.save(opt.savedir .. '/play-memory.t7', memory)
                print('Done playing!')
                print('Episodes: ', episodes, 'total reward:', totalReward)
                os.exit()
            else
                action = 2
            end
            qtimer:start()
         end)
qtimer:start()
