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
opt.rnn = opt[3]
if not opt.fpath then print('missing arg #1: missing network file to test!') return end
if not opt.gridSize then print('missing arg #2: game grid size!') return end

if opt.rnn then require 'nngraph' end

-- Initialise and start environment
local gameEnv = CatchEnvironment(opt.gridSize)

local episodes, totalReward = 0, 0


-- load trained network:
local model = torch.load(opt.fpath)

local RNNh = {} -- initial state 
if opt.rnn then
  opt.nLayers = 1
  opt.nHidden = 128
  -- Default RNN intial state set to zero:
  for l = 1, opt.nLayers do
     RNNh[l] = torch.zeros(opt.nHidden)
  end
  RNNh = table.unpack(RNNh)
  print('RNN Model is:', model.fg)
else
  print('Model is:', model)
end

-- Converts input tensor into table of dimension equal to first dimension of input tensor
-- and adds padding of zeros, which in this case are states
local function tensor2Table(inputTensor, padding)
   local outputTable = {}
   for t = 1, inputTensor:size(1) do outputTable[t] = inputTensor[t] end
   for l = 1, padding do outputTable[l + inputTensor:size(1)] = h0[l]:clone() end
   return outputTable
end


local win, reward, isGameOver, screen, action
while true do
  gameEnv:reset()
  screen = gameEnv.observe()
  isGameOver = false
  while not isGameOver do
    -- run neural net to get action:
    local netOut
    if opt.rnn then
      netOut = model:forward({ screen, RNNh } )
      RNNh = netOut[1]
      netOut = netOut[2]
    else
      netOut = model:forward(screen)
    end
    local max, index = torch.max(netOut, 1)
    action = index[1]
    
    screen, reward, isGameOver = gameEnv.act(action)
    if reward == 1 then totalReward = totalReward + reward end
    
    win = image.display({image=screen:view(opt.gridSize,opt.gridSize), zoom=10, win=win})
    -- os.execute('sleep 0.3')

    if isGameOver then
      episodes = episodes + 1
      print('Episodes: ', episodes, 'total reward:', totalReward)
    end
  end
end
