-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code

-- playing CATCH version:
-- https://github.com/SeanNaren/QlearningExample.torch

require 'CatchEnvironment'
require 'nn'
require 'image'

torch.setnumthreads(8)
torch.setdefaulttensortype('torch.FloatTensor')

local opt = {...}
opt.fpath = opt[1]
opt.gridSize = tonumber(opt[2]) 
if not opt.fpath then print('missing arg #1: missing network file to test!') return end
if not opt.gridSize then print('missing arg #2: game grid size!') return end

-- Initialise and start environment
local gameEnv = CatchEnvironment(opt.gridSize)

local episodes, totalReward = 0, 0

-- load trained network:
local model = torch.load(opt.fpath)

local win, reward, isGameOver, screen, action
while true do
  gameEnv:reset()
  screen = gameEnv.observe()
  isGameOver = false
  while not isGameOver do
    -- run neural net to get action:
    local netOut = model:forward(screen)
    local max, index = torch.max(netOut, 1)
    action = index[1]
    
    screen, reward, isGameOver = gameEnv:act(action)
    if reward == 1 then totalReward = totalReward + reward end
    
    win = image.display({image=screen:view(opt.gridSize,opt.gridSize), zoom=10, win=win})

    if isGameOver then
      episodes = episodes + 1
      print('Episodes: ', episodes, 'total reward:', totalReward)
    end
  end
end
