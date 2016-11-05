-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code

-- playing CATCH version:
-- https://github.com/SeanNaren/QlearningExample.torch

require 'torch'
require 'image'
require 'optim'
require 'nn'
require 'pl'
local of = require 'opt'
local opt = of.parse(arg)

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)

local CatchEnvironment = require 'CatchEnvironment'
local Memory = require 'memory'
local Tr = require 'train'
local tr = Tr(opt)

-- os.execute('mkdir '..opt.savedir)

local epsilon = opt.epsilon
local epsilonMinimumValue = opt.epsilonMinimumValue
local nbActions = opt.nbActions
local epoch = opt.epoch
local hiddenSize = opt.hiddenSize
local maxMemory = opt.maxMemory
local batchSize = opt.batchSize
local gridSize = opt.gridSize
local nbStates = gridSize * gridSize
local discount = opt.discount

local gameEnv = CatchEnvironment(gridSize)
local memory = Memory(maxMemory, discount)
local epsUpdate = (epsilon - epsilonMinimumValue)/epoch
local winCount = 0

print('Begin training:')
for game = 1, epoch do
    sys.tic()
    -- Initialise the environment.
    local err = 0
    gameEnv.reset()
    local isGameOver = false

    -- The initial state of the environment.
    local currentState = gameEnv.observe()

    while not isGameOver do
        local action
        -- Decides if we should choose a random action, or an action from the policy network.
        if torch.random() <= epsilon then
            action = torch.random(1, nbActions)
        else
            -- Forward the current state through the network.
            local q = tr.model:forward(currentState)
            -- Find the max index (the chosen action).
            local max, index = torch.max(q, 1)
            action = index[1]
        end

        local nextState, reward, gameOver = gameEnv.act(action)
        if (reward == 1) then winCount = winCount + 1 end
        memory.remember({
            inputState = currentState,
            action = action,
            reward = reward,
            nextState = nextState,
            gameOver = gameOver
        })
        -- Update the current state and if the game is over.
        currentState = nextState
        isGameOver = gameOver

        -- We get a batch of training data to train the model.
        local inputs, targets = memory.getBatch(tr.model, batchSize, nbActions, nbStates)

        -- Train the network which returns the error.
        err = err + tr:forward(inputs, targets)

        -- display
        if opt.display then 
            win = image.display({image=currentState:view(opt.gridSize,opt.gridSize), zoom=10, win=win})
        end
    end
    if game%opt.progFreq == 0 then 
        print(string.format("Game %d, epsilon %.2f, err = %.4f, Win count %d, Accuracy: %.2f, time [ms]: %d", 
                             game,    epsilon,      err,        winCount,     winCount/opt.progFreq, sys.toc()*1000))
        winCount = 0
    end
    -- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
    if epsilon > epsilonMinimumValue then epsilon = epsilon - epsUpdate  end
end
torch.save("catch-model-grid.net", model)
print("Model saved!")
