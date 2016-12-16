#!/usr/bin/env th

-- E. Culurciello, December 2016
-- Deep Q learning code to play game: CATCH
-- learning with RNN - version 2

require 'CatchEnvironment'
require 'nn'
require 'torch'
require 'sys'
require 'image'
require 'optim'
require 'xlua'

require 'pl'
lapp = require 'pl.lapp'
opt = lapp [[

  Game options:
  --gridSize            (default 10)         default screen resized for neural net input
  --discount            (default 0.99)       discount factor in learning
  --epsilon             (default 1)          initial value of ϵ-greedy action selection
  --epsilonMinimumValue (default 0.1)        final value of ϵ-greedy action selection
  --nbActions           (default 3)          catch number of actions
  
  Training parameters:
  --skipLearning                             skip learning and just test
  --threads               (default 8)        number of threads used by BLAS routines
  --seed                  (default 1)        initial random seed
  -r,--learningRate       (default 0.00025)  learning rate
  --batchSize             (default 64)       batch size for training
  --maxMemory             (default 1e3)      Experience Replay buffer memory
  --epochs                (default 20)       number of training steps to perform
  --learningStepsEpoch    (default 2000)     Learning steps per epoch
  --testEpisodesEpoch     (default 100)      test episodes per epoch
  --episodesWatch         (default 10)       episodes to watch after training
  --clampReward                              clamp reward to -1, 1

  -- Model parameters:
  --fw                                       Use FastWeights or not
  --nLayers               (default 1)        RNN layers
  --nHidden               (default 128)      RNN hidden size
  --nFW                   (default 8)        number of fast weights previous vectors

  Display and save parameters:
  --display                                  display stuff
  --zoom                  (default 10)       zoom display
  --saveDir          (default './results')   subdirectory to save experiments in
  --load                  (default '')       load neural network to test
]]

local rnn = require 'RNN'

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.saveDir)
print('Playing Catch game with RNN\n')

local noActionIdx = 2 -- this is the idle action: do nothing in this game
local epsilon = opt.epsilon
local nbActions = opt.nbActions
local gridSize = opt.gridSize
local nbStates = gridSize * gridSize
print('WARNING: nSeq is reset to opt.gridSize-2')
opt.nSeq = opt.gridSize-2 -- RNN sequence length in this game is grid size

-- Other parameters:
local colors = sys.COLORS


-- class ReplayMemory:
local memory = {}
local function ReplayMemory(capacity)
    local channels = 1
    memory.s = torch.zeros(capacity, opt.nSeq, nbStates) -- state
    memory.a = torch.zeros(capacity, opt.nSeq) -- action
    memory.rs = torch.zeros(capacity, opt.nHidden) -- RNN state

    memory.capacity = capacity
    memory.size = 0
    memory.pos = 1

    -- batch buffers:
    memory.bs = torch.zeros(opt.batchSize, opt.nSeq, nbStates) -- state 
    memory.ba = torch.zeros(opt.batchSize, opt.nSeq) -- action
    memory.brs = torch.zeros(opt.batchSize, opt.nHidden) -- RNN state
    
    function memory.addTransition(state, action, RNNstate)
        if memory.pos == 0 then memory.pos = 1 end -- tensors do not have 0 index items!
        memory.s[memory.pos] = state:clone()
        memory.a[memory.pos] = action:clone()
        memory.rs[memory.pos] = RNNstate:clone()
       
        memory.pos = (memory.pos + 1) % memory.capacity
        memory.size = math.min(memory.size + 1, memory.capacity)
    end

    function memory.getSample(sampleSize)
        local ri = torch.randperm(memory.size-1)
        for i=1, math.min(sampleSize,memory.size) do
            memory.bs[i] = memory.s[ri[i]]
            memory.ba[i] = memory.a[ri[i]]
            memory.brs[i] = memory.rs[ri[i]]
        end
        return memory.bs, memory.ba, memory.brs
    end

end

local sgdParams = {
    learningRate = opt.learningRate,
}

local model, criterion, prototype
local RNNh0Batch = {} -- initial state for training batches
local RNNh0Proto = {} -- initial state - prototype
local RNNhProto = {} -- state to loop through prototype in inference
-- Default RNN initial state set to zero:
for l = 1, opt.nLayers do
   RNNh0Batch[l] = torch.zeros(opt.batchSize, opt.nHidden)
   RNNh0Proto[l] = torch.zeros(1, opt.nHidden) -- prototype forward does not work on batches
   -- if opt.useGPU then RNNh0Batch[l]=RNNh0Batch[l]:cuda() RNNh0Proto[l]=RNNh0Proto[l]:cuda() end
end
RNNhProto = table.unpack(RNNh0Proto) -- initial setup of RNN prototype state

local function createNetwork(nAvailableActions)
    -- Create the base RNN model:
    -- here the state is not an image, but a vectorized version of the image
    -- next steps are convRNN models
    if opt.fw then
      print('Created fast-weights RNN with:\n- input size:', nbStates, '\n- number hidden:', opt.nHidden, 
        '\n- layers:', opt.nLayers, '\n- output size:', nbActions, '\n- sequence length:', opt.nSeq, 
        '\n- fast weights states:', opt.nFW)
      model, prototype = rnn.getModel(nbStates, opt.nHidden, opt.nLayers, nbActions, opt.nSeq, opt.nFW)
    else
      print('Created RNN with:\n- input size:', nbStates, '\n- number hidden:', opt.nHidden, 
          '\n- layers:', opt.nLayers, '\n- output size:', nbActions, '\n- sequence lenght:',  opt.nSeq)
      model, prototype = rnn.getModel(nbStates, opt.nHidden, opt.nLayers, nbActions, opt.nSeq)
    end

    -- test model:
    -- print('Testing model and prototype RNN:')
    -- local ttest 
    -- print('testing prototype RNN')
    -- if opt.useGPU then ttest = {torch.CudaTensor(1, nbStates), torch.CudaTensor(1, opt.nHidden)}
    -- else ttest = {torch.Tensor(1, nbStates), torch.Tensor(1, opt.nHidden)} end
    -- print(ttest)
    -- local a = prototype:forward(ttest)
    -- print('TEST prototype output:', a)
    -- print('Testing full model RNN:')
    -- if opt.useGPU then ttest = {torch.CudaTensor(opt.batchSize, nSeq, nbStates), torch.CudaTensor(opt.batchSize, opt.nHidden)}
    -- else ttest = {torch.Tensor(opt.batchSize, opt.nSeq, nbStates), torch.Tensor(opt.batchSize, opt.nHidden)} end
    -- print('model input:', ttest)
    -- local a = model:forward(ttest)
    -- print('TEST model output:', a)

    criterion = nn.ClassNLLCriterion()
end

-- training code:
local function learnBatch(seqs, targets, state)
    local params, gradParameters = model:getParameters()

    local function feval(x_new)
        local loss = 0
        local grOut = {}
        state = unpack(RNNh0Batch) -- NOTE: we should not do this, but somehow this give better results!!!!
        local inputs = { seqs, state } -- attach RNN states to input
        local out = model:forward(inputs)
        -- process each sequence step at a time:
        for i = 1, opt.nSeq do
            gradParameters:zero()
            loss = loss + criterion:forward(out[i], targets[{{},i}])
            grOut[i] = criterion:backward(out[i], targets[{{},i}])
        end
        table.insert(grOut, state) -- attach RNN states to grad output
        model:backward(inputs, grOut)

        return loss, gradParameters
    end

    local _, loss = optim.rmsprop(feval, params, sgdParams)
    return loss[1]
end

-- reset RNN prototype state
local function resetProtoState()
    RNNhProto = table.unpack(RNNh0Proto)
end

-- this is for online learning (single inference, batch = 1)
-- We use the prototype single cell RNN forward
-- NOTE: RNNhProto has to be initialized, afterwards is fed back in this function:
local function fwdProto(state)
    local inputs = {state:view(1, nbStates), RNNhProto}
    local q = prototype:forward(inputs)
    -- Find the max index (the chosen action):
    local max, index = torch.max(q[2][1], 1) -- [2] is the output, [1] is RNN state
    local action = index[1]
    RNNhProto = q[1] -- next prototype state feeds back to prototype
    
    return action, RNNhProto
end

-- Learns from a single transition (making use of replay memory):
local function learnFromMemory()
    -- Get a random minibatch from the replay memory and learns from it:
    if memory.size > opt.batchSize then
        local s, a, rs = memory.getSample(opt.batchSize)
        learnBatch(s, a, rs) -- states and actions are inputs and targets to learn
    end
end

local steps = 1 -- counts steps to game win
local sSeq = torch.zeros(opt.nSeq, nbStates) -- store sequence of states in successful run
local aSeq = torch.ones(opt.nSeq) -- store sequence of actions in successful run
local initRNNstate -- to store RNN state in memory
local function performLearningStep(epoch)
    -- Makes an action according to eps-greedy policy, observes the result
    -- (next state, reward) and learns from the transition

    local function explorationRate(epoch)
        --  Define exploration rate change over time:
        local start_eps = opt.epsilon
        local end_eps = opt.epsilonMinimumValue
        local const_eps_epochs = 0.1 * opt.epochs  -- 10% of learning time
        local eps_decay_epochs = 0.8 * opt.epochs  -- 60% of learning time

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

    local function resetSeqs() steps = 1 sSeq:zero() aSeq:fill(noActionIdx) end -- fill with noActionIdx

    local a, reward, gameOver, RNNstate
    local state = game.observe()
    -- With probability eps make a random action:
    local eps = explorationRate(epoch)
    -- Choose the best action according to the network:
    a, RNNstate = fwdProto(state)
    if steps == 1 then initRNNstate = RNNstate:clone() end -- save initial RNN state to be used in sequence learning!

    -- online learning: chose random action:
    if torch.uniform() <= eps then
        a = torch.random(1, nbActions)
    end
    _, reward, gameOver = game.act(a)

    -- save sequences:
    sSeq[steps] = state:clone()
    aSeq[steps] = a
    -- if it is a successful sequence, record it and the learn
    if reward > 0 then 
        -- Remember the transition that was just experienced:
        memory.addTransition(sSeq, aSeq, initRNNstate)
        -- learning step:
        learnFromMemory()
    else
        steps = steps+1
    end
    if reward ~= 0 or steps > opt.nSeq then
        -- reset step counter and sequence buffers:
        resetSeqs()
    end

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
            resetProtoState() -- reset prototype RNN state after each new game
            for learning_step=1, opt.learningStepsEpoch do
                xlua.progress(learning_step, opt.learningStepsEpoch)
                epsilon, gameOver, score = performLearningStep(epoch)
                if gameOver then
                    table.insert(trainScores, score)
                    game.reset()
                    trainEpisodesFinished = trainEpisodesFinished + 1
                    resetProtoState() -- reset prototype RNN state after each new game
                end
            end

            -- print(string.format("%d training episodes played.", trainEpisodesFinished))

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
                resetProtoState() -- reset prototype RNN state after each new game
                local r = 0
                gameOver = false
                repeat
                    local state = game.observe()
                    local bestActionIndex = fwdProto(state)
                    _, reward, gameOver = game.act(bestActionIndex)
                    r = r + reward 
                until gameOver
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
        print("Saving the network weigths to:", opt.saveDir)
            torch.save(opt.saveDir..'/proto-catch-dqn.net', prototype:clone():float():clearState())
    else
        if opt.load == '' then print('Missing neural net file to load!') os.exit() end
        prototype = torch.load(opt.load) -- otherwise load network to test!
        print('Loaded model is:', prototype)
    end 
    -- game.close()

    print("======================================")
    print("Training finished. It's time to watch!")

    for i = 1, opt.episodesWatch do
        game.reset()
        resetProtoState() -- reset prototype RNN state after each new game
        local score = 0
        local win
        gameOver = false
        repeat
            local state = game.observe()
            local bestActionIndex = fwdProto(state)
            _, reward, gameOver = game.act(bestActionIndex)
            score = score + reward
            -- display
            if opt.display then 
                win = image.display({image=state:view(opt.gridSize,opt.gridSize), zoom=opt.zoom, win=win})
            end
            sys.sleep(0.1) -- slow down game
        until gameOver

        -- Sleep between episodes:
        sys.sleep(1)
        print("Total score: ", score)
    end
    -- game.close()
end

-- run main program:
main()
