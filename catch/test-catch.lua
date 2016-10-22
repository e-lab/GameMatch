require 'nn'
local image = require 'image'
local Catch = require 'rlenvs/Catch' --install: https://github.com/Kaixhin/rlenvs

torch.setnumthreads(8)
torch.setdefaulttensortype('torch.FloatTensor')

local opt = {...}
opt.fpath = opt[1]
opt.gridSize = tonumber(opt[2]) 
if not opt.fpath then print('missing arg #1: missing network file to test!') return end
if not opt.gridSize then print('missing arg #2: game grid size!') return end

-- Initialise and start environment
local gameEnv = Catch({size = opt.gridSize, level = 1})
local stateSpec = gameEnv:getStateSpec()
local actionSpec = gameEnv:getActionSpec()
local screen = gameEnv:start()
local gameActions = {0,1,2} -- game actions from CATCH

print('State size is:', screen:size())

local reward, terminal
local episodes, totalReward = 0, 0

-- load trained network:
local model = torch.load(opt.fpath)

-- simple state: just get ball X,Y and paddle X position and concatenate these values!
local function getSimpleState(inState)
  local val, ballx, bally, paddlex1
  -- print(inState)
  bally = inState[{{},{1,opt.gridSize-1},{}}]:max(2):squeeze()
  ballx = inState[{{},{1,opt.gridSize-1},{}}]:max(3):squeeze()
  paddlex1 = inState[{{},{opt.gridSize},{}}]:max(2):squeeze()
  -- print(ballx, bally, paddlex1)
  local out = torch.cat(ballx, bally)
  out = torch.cat(out, paddlex1)
  -- print(out)
  -- io.read()
  return out
end

-- Display
local win = image.display({image=screen, zoom=10})

while true do
  -- run neural net to get action:
  local state = getSimpleState(screen)
  local netOut = model:forward(state)
  local max, index = torch.max(netOut, 1)
  local action = index[1]
  
  reward, screen, terminal = gameEnv:step(gameActions[action])
  totalReward = totalReward + reward
  
  win = image.display({image=screen, zoom=10, win=win})

  -- If game finished, start again
  if terminal then
    episodes = episodes + 1
    screen = gameEnv:start()
    print('Episodes: ', episodes, 'total reward:', totalReward)
  end
end
