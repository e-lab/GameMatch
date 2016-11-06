-- Eugenio Culurciello
-- October 2016
-- playing CATCH version / human player
-- 
-- loosely based on: https://github.com/SeanNaren/QlearningExample.torch

require 'CatchEnvironment'
require 'nn'
require 'image'

require 'qtwidget' -- for keyboard interaction

torch.setnumthreads(8)
torch.setdefaulttensortype('torch.FloatTensor')

local qtimer = qt.QTimer()

local opt = {...}
opt.fpath = opt[1]
opt.gridSize = tonumber(opt[2]) or 10
-- if not opt.fpath then print('missing arg #1: missing file to save to!') return end
-- if not opt.gridSize then print('missing arg #2: game grid size!') return end
opt.zoom = 10
local nbActions = 3 -- in catch
local gridSize = opt.gridSize
local nbStates = gridSize * gridSize
local nSeq = gridSize-2 -- RNN sequence length in this game is grid size

-- Initialise and start environment
local gameEnv = CatchEnvironment(opt.gridSize)

local episodes, totalReward = 0, 0

-- Create a window for displaying output frames
local win = qtwidget.newwindow(opt.zoom*opt.gridSize, opt.zoom*opt.gridSize,'EC game engine')

local reward, isGameOver, currentState
local action = 2
local seqMem = torch.Tensor(nSeq, nbStates) -- store sequence of states in successful run
local seqAct = torch.zeros(nSeq, nbActions) -- store sequence of actions in successful run
local memory = {} -- memory to save play data

gameEnv.reset()
isGameOver = false
currentState = gameEnv.observe()
local steps = 0 -- count steps to game win

function main()
  steps = steps+1
  -- look at screen:
  win = image.display({image=currentState:view(opt.gridSize,opt.gridSize), zoom=opt.zoom, win=win})

  -- get human player move:
  currentState, reward, isGameOver = gameEnv.act(action)

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

-- game controls
print('Catch game controls: arrow keys left / right')
qtimer.interval = 300
qtimer.singleShot = false
qt.connect(qtimer,'timeout()', main)

qt.connect(win.listener,
         'sigKeyPress(QString, QByteArray, QByteArray)',
         function(_, keyValue)
            if keyValue == 'Key_Right' then
                action = 3
            elseif keyValue == 'Key_Left' then
                action = 1
            elseif keyValue == 'Key_Q' then
                torch.save('play-memory.t7', memory)
                print('Done playing!')
                print('Episodes: ', episodes, 'total reward:', totalReward)
                os.exit()
            else
                action = 2
            end
         end)
qtimer:start()
