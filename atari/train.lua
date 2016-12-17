-- Eugenio Culurciello
-- October-December 2016
-- Deep RNN for reinforcement online learning on Atari games / breakout

require 'initenv' -- Atari game environment
require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'pl'
lapp = require 'pl.lapp'
opt = lapp [[

  Game options:
  --gridSize            (default 20)          game grid size 
  --discount            (default 0.99)        discount factor in learning
  --epsilon             (default 1)           initial value of ϵ-greedy action selection
  --epsilonMinimumValue (default 0.1)         final value of ϵ-greedy action selection
  --playFile            (default '')          human play file to initialize exp. replay memory
  --framework           (default 'alewrap')         name of training framework
  --env                 (default 'breakout')        name of environment to use')
  --game_path           (default 'roms/')           path to environment file (ROM)
  --env_params          (default 'useRGB=true')     string of environment parameters
  --pool_frms_type      (default "max")             pool inputs frames mode
  --pool_frms_size      (default "1")               pool inputs frames size
  --actrep              (default 4)                 how many times to repeat action, frames to skip to speed up game and inference
  --randomStarts        (default 30)                play action 0 between 1 and random_starts number of times at the start of each training episode

  Training parameters:
  --skipLearning                              skip learning and just test
  --threads               (default 8)         number of threads used by BLAS routines
  --seed                  (default 1)         initial random seed
  -r,--learningRate       (default 0.00025)   learning rate
  --batchSize             (default 64)        batch size for training
  --maxMemory             (default 1e4)       Experience Replay buffer memory
  --epochs                (default 100)       number of training steps to perform
  --learningStepsEpoch    (default 5000)      Learning steps per epoch
  --testEpisodesEpoch     (default 100)       test episodes per epoch
  --episodesWatch         (default 10)        episodes to watch after training
  --clampReward                               clamp reward to -1, 1
  --useGPU                                    use GPU in training
  --gpuId                 (default 1)         which GPU to use

  Display and save parameters:
  --zoom                  (default 4)         zoom window
  --display                                   display stuff
  --saveDir          (default './results')    subdirectory to save experiments in
  --load                  (default '')        load neural network to test
]]
   
-- format options:
opt.pool_frms = 'type=' .. opt.pool_frms_type .. ',size=' .. opt.pool_frms_size

print('Options are:', opt)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.saveDir)
torch.save(paths.concat(opt.saveDir,'opt.t7'),opt)

-- Other parameters:
local colors = sys.COLORS

-- setup game environment:
local game, gameActions, agent, opt = gameEnvSetup(opt)
print(string.format(colors.green..'Atari game environment started. Number of game actions: %d', #gameActions))
local epsilon = opt.epsilon
local nbActions = #gameActions
local gridSize = opt.gridSize
local nbStates = gridSize * gridSize

-- Converts and down-samples the input image:
local poolnet = nn.SpatialMaxPooling(4,4,4,4)
local function screenPreProcess(inImage)
  -- this is the general case: whole screen:
  -- return image.scale(inImage[1], gridSize, gridSize, 'simple'):sum(1):div(3) -- we also convert to B/W from color
  -- this is a simple version looking just at the inside grid:
  local pooled = poolnet:forward(inImage[1][{{},{94,194},{9,152}}])
  local outImage = image.scale(pooled, opt.gridSize, opt.gridSize):sum(1):div(3)
  return outImage
end

-- class ReplayMemory:
local memory = {}
local function ReplayMemory(capacity)
    local channels = 1 -- number of image planes
    memory.s1 = torch.zeros(capacity, channels, gridSize, gridSize)
    memory.s2 = torch.zeros(capacity, channels, gridSize, gridSize)
    memory.a = torch.ones(capacity)
    memory.r = torch.zeros(capacity)
    memory.isterminal = torch.zeros(capacity)

    memory.capacity = capacity
    memory.size = 0
    memory.pos = 1

    -- batch buffers:
    memory.bs1 = torch.zeros(opt.batchSize, channels, gridSize, gridSize)
    memory.bs2 = torch.zeros(opt.batchSize, channels, gridSize, gridSize)
    memory.ba = torch.ones(opt.batchSize)
    memory.br = torch.zeros(opt.batchSize)
    memory.bisterminal = torch.zeros(opt.batchSize)

    function memory.addTransition(s1, action, s2, isterminal, reward)
        if memory.pos == 0 then memory.pos = 1 end -- tensors do not have 0 index items!
        memory.s1[{memory.pos, {}}] = s1:clone()
        memory.a[memory.pos] = action
        if not isterminal then
            memory.s2[{memory.pos, {}}] = s2:clone()
        end
        memory.isterminal[memory.pos] = isterminal and 1 or 0 -- boolean stored as 0 or 1 in memory!
        memory.r[memory.pos] = reward

        memory.pos = (memory.pos + 1) % memory.capacity
        memory.size = math.min(memory.size + 1, memory.capacity)
    end

    function memory.getSample(sampleSize)
        for i=1,sampleSize do
            local ri = torch.random(1, memory.size-1)
            memory.bs1[i] = memory.s1[ri]:clone()
            memory.bs2[i] = memory.s2[ri]:clone()
            memory.ba[i] = memory.a[ri]
            memory.bisterminal[i] = memory.isterminal[ri]
            memory.br[i] = memory.r[ri]
        end
        return memory.bs1, memory.ba, memory.bs2, memory.bisterminal, memory.br
    end

end

local sgdParams = {
    learningRate = opt.learningRate,
}

local model, criterion
local function createNetwork(nAvailableActions)

    -- Create the base model:
    model = nn.Sequential()
    model:add(nn.SpatialConvolution(1,32,8,8,4,4))
    model:add(nn.SpatialConvolution(32,64,4,4,2,2))
    model:add(nn.View(64))
    model:add(nn.Linear(64, nbActions))
    -- test:
    local retvt = model:forward(torch.Tensor(1, gridSize, gridSize)) --test
    -- print('test model:', retvt)
    print('Model to train:', model)

    criterion = nn.MSECriterion()
end

local function learnBatch(state, targets)

    local params, gradParams = model:getParameters()
    
    local function feval(x_new)
        gradParams:zero()
        local predictions = model:forward(state)
        local loss = criterion:forward(predictions, targets)
        local gradOutput = criterion:backward(predictions, targets)
        model:backward(state, gradOutput)
        return loss, gradParams
    end

    local _, fs = optim.rmsprop(feval, params, sgdParams)
    return fs[1] -- loss
end

local function getQValues(state)
    return model:forward(state)
end

local function getBestAction(state)
    local q = getQValues(state)
    local max, index = torch.max(q, 1)
    local action = index[1]
    return action, q
end

local function learnFromMemory()
    -- Learns from a single transition (making use of replay memory)
    -- s2 is ignored if s2_isterminal

    -- Get a random minibatch from the replay memory and learns from it:
    if memory.size > opt.batchSize then
        local s1, a, s2, isterminal, r = memory.getSample(opt.batchSize)
        if opt.clampReward then r = r:clamp(-1,1) end -- clamping of reward!

        local q2 = torch.max(getQValues(s2), 2) -- get max q for each sample of batch
        local target_q = getQValues(s1):clone()

        -- target differs from q only for the selected action. The following means:
        -- target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r:
        for i=1,opt.batchSize do
            target_q[i][a[i]] = r[i] + opt.discount * (1 - isterminal[i]) * q2[i]
        end
        learnBatch(s1, target_q)
    end
end

local function performLearningStep(epoch)
    -- Makes an action according to eps-greedy policy, observes the result
    -- (next state, reward) and learns from the transition

    local function explorationRate(epoch)
        --  Define exploration rate change over time:
        local start_eps = opt.epsilon
        local end_eps = opt.epsilonMinimumValue
        local const_eps_epochs = 0.1 * opt.epochs  -- 10% of learning time
        local eps_decay_epochs = 0.8 * opt.epochs  -- 80% of learning time

        if epoch < const_eps_epochs then
            return start_eps
        elseif epoch < eps_decay_epochs then
            -- Linear decay:
            return start_eps - (epoch - const_eps_epochs) /
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else
            return end_eps
        end
    end

    local screen = game:step(gameActions[1], true) -- just step once to start!
    local s1 = screenPreProcess(screen)

    -- With probability eps make a random action:
    local eps = explorationRate(epoch)
    local a, s2, reward, gameOver
    if torch.uniform() <= eps then
        a = torch.random(1, nbActions)
    else
        -- Choose the best action according to the network:
        a = getBestAction(s1)
    end
    screen, reward, gameOver = game:step(gameActions[a], true)
    s2 = screenPreProcess(screen)

    if gameOver then s2 = nil end

    if opt.display then 
      win = image.display({image=s1, zoom=opt.zoom, win=win})
    end

    -- Remember the transition that was just experienced:
    memory.addTransition(s1, a, s2, gameOver, reward)

    learnFromMemory()

    return eps, gameOver, reward
end

local logger = optim.Logger(opt.saveDir..'/model-atari-dqn.log')
logger:setNames{'Training acc. %', 'Test acc. %'} -- log train / test accuracy in percent [%]

local win -- window for display
local function main()
    local epsilon, gameOver, score, reward, screen

    local timeStart = sys.tic()
    if not opt.skipLearning then
        -- Create replay memory which will store the play data:
        ReplayMemory(opt.maxMemory)
        createNetwork(nbActions)
        
        print("Starting the training!")
        for epoch = 1, opt.epochs do
            print(string.format(colors.green.."\nEpoch %d\n-------", epoch))
            local trainEpisodesFinished = 0
            local trainScores = {}

            print(colors.red.."Training...")
            game:nextRandomGame()
            score = 0 
            for learningStep = 1, opt.learningStepsEpoch do
                xlua.progress(learningStep, opt.learningStepsEpoch)
                epsilon, gameOver, reward = performLearningStep(epoch)
                score = score + reward
                if gameOver then
                    table.insert(trainScores, score)
                    score = 0 
                    game:nextRandomGame()
                    trainEpisodesFinished = trainEpisodesFinished + 1
                    collectgarbage()
                end
            end

            print(string.format("%d training episodes played.", trainEpisodesFinished))

            trainScores = torch.Tensor(trainScores)

            -- print(string.format("Results: mean: %.1f, std: %.1f, min: %.1f, max: %.1f", 
                -- trainScores:mean(), trainScores:std(), trainScores:min(), trainScores:max()))
            local logTrain = trainScores:gt(0):sum()/trainEpisodesFinished*100
            print(string.format("Games played: %d, Accuracy: %d %%", trainEpisodesFinished, logTrain))
            print('Epsilon value', epsilon)

            -- print(colors.red.."\nTesting...")
            -- local testEpisode = {}
            -- local testScores = {}
            -- for testEpisode = 1, opt.testEpisodesEpoch do
            --     xlua.progress(testEpisode, opt.testEpisodesEpoch)
            --     screen = game:nextRandomGame()
            --     score = 0
            --     gameOver = false
            --     while not gameOver do
            --         local state = screenPreProcess(screen)
            --         local bestActionIndex = getBestAction(state)
            --         state, reward, gameOver = game:step(gameActions[bestActionIndex], true)
            --         score = score + reward
            --         if opt.display then 
            --           win = image.display({image=screen, zoom=opt.zoom, win=win})
            --         end
            --     end
            --     collectgarbage()
            --     table.insert(testScores, score)
            -- end

            -- testScores = torch.Tensor(testScores)
            -- print(string.format("Results: mean: %.1f, std: %.1f, min: %.1f, max: %.1f",
            --     testScores:mean(), testScores:std(), testScores:min(), testScores:max()))
            -- local logTest = testScores:gt(0):sum()/opt.testEpisodesEpoch*100
            -- print(string.format("Games played: %d, Accuracy: %d %%", 
            --     opt.testEpisodesEpoch, logTest))
            
            -- print(string.format(colors.cyan.."Total elapsed time: %.2f minutes", sys.toc()/60.0))
            -- logger:add{ logTrain, logTest }
            collectgarbage()
        end
    else
        if opt.load == '' then print('Missing neural net file to load!') os.exit() end
        model = torch.load(opt.load) -- otherwise load network to test!
        print('Loaded model is:', model)
    end 
    print("Saving the network weigths to:", opt.saveDir)
            torch.save(opt.saveDir..'/model-atari-dqn.net', model:clone():float():clearState())
    -- game:close()

    print("======================================")
    print("Training finished. It's time to watch!")

    for i = 1, opt.episodesWatch do
        screen = game:nextRandomGame()
        score = 0
        gameOver = false
        while not gameOver do
            local state = screenPreProcess(screen)
            local action = getBestAction(state)
            -- play game in test mode (episodes don't end when losing a life)
            state, reward, gameOver = game:step(gameActions[action], false)
            score = score + reward
            -- display
            if opt.display then 
                win = image.display({image=screen, zoom=opt.zoom, win=win})
            end
            sys.sleep(0.1) -- slow down game
        end

        -- Sleep between episodes:
        sys.sleep(1)
        print("Total score: ", score)
    end
    -- game:close()
end

-- run main program:
main()
