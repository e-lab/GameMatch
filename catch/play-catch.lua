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

-- Initialise and start environment
local gameEnv = CatchEnvironment(opt.gridSize)

local episodes, totalReward = 0, 0

-- Create a window for displaying output frames
local win = qtwidget.newwindow(opt.zoom*opt.gridSize, opt.zoom*opt.gridSize,'EC game engine')

local reward, isGameOver, screen
local keyPress = 2


gameEnv.reset()
isGameOver = false
screen = gameEnv.observe()

function main()
  -- look at screen:
  win = image.display({image=screen:view(opt.gridSize,opt.gridSize), zoom=opt.zoom, win=win})

  -- get human player move:
  screen, reward, isGameOver = gameEnv.act(keyPress)
  keyPress = 2 -- stop move
  
  if reward == 1 then 
    totalReward = totalReward + reward
    print('Total Reward:', totalReward)
  end

  if isGameOver then
    episodes = episodes + 1
    -- print('Episodes: ', episodes, 'total reward:', totalReward)
    gameEnv.reset()
    isGameOver = false
  end

end

-- game controls
print('Game controls: left / right')
qtimer.interval = 300
qtimer.singleShot = false
qt.connect(qtimer,'timeout()', main)

local prevState = true
qt.connect(win.listener,
         'sigKeyPress(QString, QByteArray, QByteArray)',
         function(_, keyValue)
            if keyValue == 'Key_Right' then
                keyPress = 0
            elseif keyValue == 'Key_Left' then
                keyPress = 1
            elseif keyValue == 'Key_Q' then
                -- torch.save()
                print('Done playing!')
                os.exit()
            else
                keyPress = 2
            end
            qtimer:start()
         end)
qtimer:start()
