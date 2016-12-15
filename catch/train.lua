-- Eugenio Culurciello
-- October-Dember 2016
-- Deep Q learning code to play game: CATCH

require 'CatchEnvironment'
require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'pl'
lapp = require 'pl.lapp'
opt = lapp [[
  
  Game options:
  --gridSize            (default 10)          state is screen resized to this size 
  --discount            (default 0.99)        discount factor in learning
  --epsilon             (default 1)           initial value of ϵ-greedy action selection
  --epsilonMinimumValue (default 0.1)         final value of ϵ-greedy action selection
  --nbActions           (default 3)           catch number of actions
  
  Training parameters:
  --skipLearning                             skip learning and just test
  --threads               (default 8)        number of threads used by BLAS routines
  --seed                  (default 1)        initial random seed
  -r,--learningRate       (default 0.00025)  learning rate
  --batchSize             (default 64)       batch size for training
  --maxMemory             (default 1e3)      Experience Replay buffer memory
  --epochs                (default 10)       number of training steps to perform
  --learningStepsEpoch    (default 1000)     Learning steps per epoch
  --testEpisodesEpoch     (default 100)      test episodes per epoch
  --episodesWatch         (default 10)       episodes to watch after training
  --clampReward                              clamp reward to -1, 1

  Model parameters:
  --modelType             (default 'mlp')    neural net model type: cnn, mlp
  --nHidden               (default 128)      hidden states in neural net

  Display and save parameters:
  --zoom                  (default 10)       zoom window
  --display                                  display stuff
  --saveDir          (default './results')   subdirectory to save experiments in
  --load                  (default '')       load neural network to test
]]

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.saveDir)
print('Playing Catch game with Q-learning and mlp/cnn\n')

local epsilon = opt.epsilon
local nbActions = opt.nbActions
local gridSize = opt.gridSize
local nbStates = gridSize * gridSize

-- Other parameters:
local colors = sys.COLORS


-- class ReplayMemory:
local memory = {}
local function ReplayMemory(capacity)
    memory.s1 = torch.zeros(capacity, nbStates)
    memory.s2 = torch.zeros(capacity, nbStates)
    memory.a = torch.ones(capacity)
    memory.r = torch.zeros(capacity)
    memory.isterminal = torch.zeros(capacity)

    memory.capacity = capacity
    memory.size = 0
    memory.pos = 1

    -- batch buffers:
    memory.bs1 = torch.zeros(opt.batchSize, nbStates)
    memory.bs2 = torch.zeros(opt.batchSize, nbStates)
    memory.ba = torch.ones(opt.batchSize)
    memory.br = torch.zeros(opt.batchSize)
    memory.bisterminal = torch.zeros(opt.batchSize)

    function memory.addTransition(s1, action, s2, isterminal, reward)
        if memory.pos == 0 then memory.pos = 1 end -- tensors do not have 0 index items!
        memory.s1[{memory.pos, {}}] = s1
        memory.a[memory.pos] = action
        if not isterminal then
            memory.s2[{memory.pos, {}}] = s2
        end
        memory.isterminal[memory.pos] = isterminal and 1 or 0 -- boolean stored as 0 or 1 in memory!
        memory.r[memory.pos] = reward

        memory.pos = (memory.pos + 1) % memory.capacity
        memory.size = math.min(memory.size + 1, memory.capacity)
    end

    function memory.getSample(sampleSize)
        for i=1,sampleSize do
            local ri = torch.random(1, memory.size-1)
            memory.bs1[i] = memory.s1[ri]
            memory.bs2[i] = memory.s2[ri]
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
    if opt.modelType == 'mlp' then
        model:add(nn.Linear(nbStates, opt.nHidden))
        model:add(nn.ReLU())
        model:add(nn.Linear(opt.nHidden, opt.nHidden))
        model:add(nn.ReLU())
        model:add(nn.Linear(opt.nHidden, nbActions))
        -- test:
        local retvt = model:forward(torch.Tensor(nbStates))
        -- print('test model:', retvt)
    elseif opt.modelType == 'cnn' then
        model:add(nn.View(1, opt.gridSize, opt.gridSize))
        model:add(nn.SpatialConvolution(1, 32, 4,4, 2,2))
        model:add(nn.ReLU())
        model:add(nn.View(32*16))
        model:add(nn.Linear(32*16, opt.nHidden))
        model:add(nn.ReLU())
        model:add(nn.Linear(opt.nHidden, nbActions))
        -- test:
        local retvt = model:forward(torch.Tensor(gridSize, gridSize))
        -- print('test model:', retvt)
    else
        print('model type not recognized')
    end
    print('Model '..opt.modelType.. ' is:', model)

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

    local s1 = game.observe()

    -- With probability eps make a random action:
    local eps = explorationRate(epoch)
    local a, s2, reward, gameOver
    if torch.uniform() <= eps then
        a = torch.random(1, nbActions)
    else
        -- Choose the best action according to the network:
        a = getBestAction(s1)
    end
    s2, reward, gameOver = game.act(a)

    if gameOver then s2 = nil end

    -- Remember the transition that was just experienced:
    memory.addTransition(s1, a, s2, gameOver, reward)

    learnFromMemory()

    return eps, gameOver, reward
end


-- Create Catch game instance:
game = CatchEnvironment(gridSize)
print("Catch game initialized.")

local logger = optim.Logger(opt.saveDir..'/model-catch-dqn.log')
logger:setNames{'Training acc. %', 'Test acc. %'} -- log train / test accuracy in percent [%]

local function main()
    local epsilon, gameOver, score, reward

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
            game.reset()
            for learning_step=1, opt.learningStepsEpoch do
                xlua.progress(learning_step, opt.learningStepsEpoch)
                epsilon, gameOver, score = performLearningStep(epoch)
                if gameOver then
                    table.insert(trainScores, score)
                    game.reset()
                    trainEpisodesFinished = trainEpisodesFinished + 1
                end
            end

            print(string.format("%d training episodes played.", trainEpisodesFinished))

            trainScores = torch.Tensor(trainScores)

            -- print(string.format("Results: mean: %.1f, std: %.1f, min: %.1f, max: %.1f", 
                -- trainScores:mean(), trainScores:std(), trainScores:min(), trainScores:max()))
            local logTrain = trainScores:gt(0):sum()/trainEpisodesFinished*100
            print(string.format("Games played: %d, Accuracy: %d %%", trainEpisodesFinished, logTrain))
            print('Epsilon value', epsilon)

            print(colors.red.."\nTesting...")
            local testEpisode = {}
            local testScores = {}
            for testEpisode=1, opt.testEpisodesEpoch do
                xlua.progress(testEpisode, opt.testEpisodesEpoch)
                game.reset()
                local r = 0
                gameOver = false
                while not gameOver do
                    local state = game.observe()
                    local bestActionIndex = getBestAction(state)
                    _, reward, gameOver = game.act(bestActionIndex)
                    r = r + reward 
                end
                table.insert(testScores, r)
            end

            testScores = torch.Tensor(testScores)
            -- print(string.format("Results: mean: %.1f, std: %.1f, min: %.1f, max: %.1f",
                -- testScores:mean(), testScores:std(), testScores:min(), testScores:max()))
            local logTest = testScores:gt(0):sum()/opt.testEpisodesEpoch*100
            print(string.format("Games played: %d, Accuracy: %d %%", 
                opt.testEpisodesEpoch, logTest))
            
            print(string.format(colors.cyan.."Total elapsed time: %.2f minutes", sys.toc()/60.0))
            logger:add{ logTrain, logTest }
        end
    else
        if opt.load == '' then print('Missing neural net file to load!') os.exit() end
        model = torch.load(opt.load) -- otherwise load network to test!
        print('Loaded model is:', model)
    end 
    print("Saving the network weigths to:", opt.saveDir)
            torch.save(opt.saveDir..'/model-catch-dqn.net', model:clone():float():clearState())
    -- game:close()

    print("======================================")
    print("Training finished. It's time to watch!")

    for i = 1, opt.episodesWatch do
        game.reset()
        local score = 0
        local win
        gameOver = false
        while not gameOver do
            local state = game.observe()
            local bestActionIndex = getBestAction(state)
            _, reward, gameOver = game.act(bestActionIndex)
            score = score + reward
            -- display
            if opt.display then 
                win = image.display({image=state:view(opt.gridSize,opt.gridSize), zoom=opt.zoom, win=win})
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
