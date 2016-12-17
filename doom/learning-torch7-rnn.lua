#!/usr/bin/env th

-- E. Culurciello, December 2016
-- based on https://github.com/Marqt/ViZDoom/blob/master/examples/python/learning_tensorflow.py
-- use Recurrent Neural networks (RNN) to train simple Doom scenarios

local base_path="/Users/eugenioculurciello/Desktop/ViZDoom/"
package.path = package.path .. ";"..base_path.."lua/vizdoom/?.lua"
require 'vizdoom.init'
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
  --gridSize            (default 30)         default screen resized for neural net input
  --discount            (default 0.99)       discount factor in learning
  --epsilon             (default 1)          initial value of ϵ-greedy action selection
  --epsilonMinimumValue (default 0.1)        final value of ϵ-greedy action selection
  
  Training parameters:
  --skipLearning                             skip learning and just test
  --threads               (default 8)        number of threads used by BLAS routines
  --seed                  (default 1)        initial random seed
  -r,--learningRate       (default 0.001)    learning rate
  --batchSize             (default 64)       batch size for training
  --maxMemory             (default 1e4)      Experience Replay buffer memory
  --epochs                (default 20)       number of training steps to perform

  -- Q-learning settings:
  --learningStepsEpoch    (default 2000)     Learning steps per epoch
  --clampReward                              clamp reward to -1, 1

  -- Model parameters:
  --nSeq                  (default 100)      RNN maximum sequence length
  --fw                                       Use FastWeights or not
  --nLayers               (default 1)        RNN layers
  --nHidden               (default 128)      RNN hidden size
  --nFW                   (default 8)        number of fast weights previous vectors

  -- Training regime
  --testEpisodesEpoch     (default 100)      test episodes per epoch
  --frameRepeat           (default 12)       repeat frames / actions N times
  --episodesWatch         (default 10)       episodes to watch after training
  
  Display and save parameters:
  --display                                  display stuff
  --zoom                  (default 4)        zoom for display
  --saveDir          (default './results')   subdirectory to save experiments in
  --load                  (default '')       load neural network to test
]]

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.saveDir)
print('Playing Doom game with RNN\n')

-- RNN package:
package.path = '../catch/?.lua;' .. package.path
local rnn = require 'RNN'

-- Other parameters:
local noActionIdx = 2 -- this is the idle action: do nothing in this game
local resolution = {opt.gridSize*1.5, opt.gridSize} -- Y, X sizes of rescaled state / game screen
local nbStates = resolution[1]*resolution[2] -- size of RNN input vector (game state treated as vector here)

local colors = sys.COLORS

-- Configuration file path:
local config_file_path = base_path.."scenarios/simpler_basic.cfg"
-- local config_file_path = base_path.."scenarios/rocket_basic.cfg"
-- local config_file_path = base_path.."scenarios/basic.cfg"

-- Doom basic scenario actions:
local actions = {
    [1] = torch.Tensor({1,0,0}),
    [2] = torch.Tensor({0,1,0}),
    [3] = torch.Tensor({0,0,1})
}


-- Converts and down-samples the input image
local function preprocess(inImage)
    inImage = inImage:float():div(255)
    return image.scale(inImage, unpack(resolution))
end

-- class ReplayMemory:
local memory = {}
local function ReplayMemory(capacity)
    local channels = 1
    memory.s = torch.zeros(capacity, opt.nSeq, channels*resolution[1]*resolution[2]) -- state
    memory.a = torch.zeros(capacity, opt.nSeq) -- action
    memory.rs = torch.zeros(capacity, opt.nHidden) -- RNN state

    memory.capacity = capacity
    memory.size = 0
    memory.pos = 1

    -- batch buffers:
    memory.bs = torch.zeros(opt.batchSize, opt.nSeq, channels*resolution[1]*resolution[2])
    memory.ba = torch.zeros(opt.batchSize, opt.nSeq)
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

local function createNetwork(nbActions)
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
    -- Find the max index (the chosen action).
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

local function shiftSeq(s, a, steps)
    local sn = s:clone()
    local an = a:clone()
    local shift = opt.nSeq - steps
    for i=1, opt.nSeq do
        sn[i] = s[(i-shift-1)%opt.nSeq+1]
        an[i] = a[(i-shift-1)%opt.nSeq+1]
    end
    return sn, an
end

local win -- window for displaying results
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

    local function resetSeqs() steps = 1 sSeq:zero() aSeq:fill(noActionIdx) end -- fill with noActionIdx

    local a, reward, gameOver, RNNstate
    local state = preprocess(game:getState().screenBuffer)

    if opt.display then win=image.display({image=state, win=win, zoom=opt.zoom}) end

    -- With probability eps make a random action:
    local eps = explorationRate(epoch)
    -- Choose the best action according to the network:
    local a, RNNstate = fwdProto(state)
    if steps == 1 then initRNNstate = RNNstate:clone() end -- save initial RNN state to be used in sequence learning!
    
    -- online learning: chose random action:
    if torch.uniform() <= eps then
        a = torch.random(1, #actions)
    end
    local reward = game:makeAction(actions[a], opt.frameRepeat)

    -- save sequences:
    sSeq[steps] = state:clone()
    aSeq[steps] = a
    
    -- if it is a successful sequence, record it and the learn
    if reward > 0 then 
        -- shift sequence so end of game is last item in list:
        sSeq, aSeq = shiftSeq(sSeq, aSeq, steps)
        -- Remember the transition that was just experienced:
        memory.addTransition(sSeq, aSeq, initRNNstate)
        -- learning step:
        learnFromMemory()
        -- reset step counter and sequence buffers:
        resetSeqs()
    else 
        steps = steps+1
    end
    if steps > opt.nSeq then
        -- reset step counter and sequence buffers:
        resetSeqs()
    end
    
    return eps,  gameOver, reward
end

-- Creates and initializes ViZDoom environment:
local function initializeViZdoom(config_file_path)
    print("Initializing doom...")
    local game = vizdoom.DoomGame()
    game:setViZDoomPath(base_path.."bin/vizdoom")
    game:setDoomGamePath(base_path.."scenarios/freedoom2.wad")
    game:loadConfig(config_file_path)
    game:setWindowVisible(opt.display)
    game:setMode(vizdoom.Mode.PLAYER)
    game:setScreenFormat(vizdoom.ScreenFormat.GRAY8)
    game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)
    game:init()
    print("Doom initialized.")
    return game
end


local logger = optim.Logger(opt.saveDir..'/-doom-rnn.log')
logger:setNames{'Training acc. %', 'Test acc. %'} -- log train / test accuracy in percent [%]

local function main()
    local epsilon, logTrain, logTest

    -- Create Doom instance:
    game = initializeViZdoom(config_file_path)

    -- Action = which buttons are pressed:
    local n = game:getAvailableButtonsSize()
    
    local time_start = sys.tic()
    -- Create replay memory which will store the play data:
    if not opt.skipLearning then
        ReplayMemory(opt.maxMemory)
        createNetwork(#actions)
    
        print("Starting the training!")
        for epoch = 1, opt.epochs do
            print(string.format(colors.green.."\nEpoch %d\n-------", epoch))
            local trainEpisodesFinished = 0
            local trainScores = {}

            print(colors.red.."Training...")
            game:newEpisode()
            resetProtoState() -- reset prototype RNN state after each new game
            for learningStep=1, opt.learningStepsEpoch do
                xlua.progress(learningStep, opt.learningStepsEpoch)
                epsilon = performLearningStep(epoch)
                if game:isEpisodeFinished() then
                    local score = game:getTotalReward()
                    table.insert(trainScores, score)
                    game:newEpisode()
                    trainEpisodesFinished = trainEpisodesFinished + 1
                    resetProtoState() -- reset prototype RNN state after each new game
                    collectgarbage()
                end
            end

            -- print(string.format("%d training episodes played.", trainEpisodesFinished))

            trainScores = torch.Tensor(trainScores)
            logTrain = trainScores:gt(0):sum()/trainEpisodesFinished*100
            print(string.format("Results: mean: %.1f, std: %.1f, min: %.1f, max: %.1f", 
                trainScores:mean(), trainScores:std(), trainScores:min(), trainScores:max()))
            print(string.format("Games played: %d, Accuracy: %d %%", trainEpisodesFinished, logTrain))
            print('Epsilon value', epsilon)

            if epoch > 2 then 
                print(colors.red.."\nTesting...")
                local testEpisode = {}
                local testScores = {}
                for testEpisode=1, opt.testEpisodesEpoch do
                    xlua.progress(testEpisode, opt.testEpisodesEpoch)
                    game:newEpisode()
                    resetProtoState() -- reset prototype RNN state after each new game
                    while not game:isEpisodeFinished() do
                        local state = preprocess(game:getState().screenBuffer)
                        local bestActionIndex = fwdProto(state)
                        game:makeAction(actions[bestActionIndex], opt.frameRepeat)
                    end
                    local r = game:getTotalReward()
                    table.insert(testScores, r)
                    collectgarbage()
                end

                testScores = torch.Tensor(testScores)
                logTest = testScores:gt(0):sum()/opt.testEpisodesEpoch*100
                print(string.format("Results: mean: %.1f, std: %.1f, min: %.1f, max: %.1f",
                    testScores:mean(), testScores:std(), testScores:min(), testScores:max()))
                print(string.format("Games played: %d, Accuracy: %d %%", opt.testEpisodesEpoch, logTest))
            end

            print(string.format(colors.cyan.."Total elapsed time: %.2f minutes", sys.toc()/60.0))
            logger:add{ logTrain, logTest }
            collectgarbage()
        end
        print("Saving the network weigths to:", opt.saveDir)
            torch.save(opt.saveDir..'/proto-doom-rnn.net', prototype:clone():float():clearState())
    else
        if opt.load == '' then print('Missing neural net file to load!') os.exit() end
        prototype = torch.load(opt.load) -- otherwise load network to test!
        print('Loaded model is:', prototype)
    end
    
    game:close()
    print("======================================")
    print("Training finished. It's time to watch!")

    -- Reinitialize the game with window visible:
    game:setWindowVisible(true)
    game:setMode(vizdoom.Mode.ASYNC_PLAYER)
    game:init()

    for i = 1, opt.episodesWatch do
        game:newEpisode()
        resetProtoState() -- reset prototype RNN state after each new game
        while not game:isEpisodeFinished() do
            local state = preprocess(game:getState().screenBuffer)
            local bestActionIndex = fwdProto(state)

            game:makeAction(actions[bestActionIndex])
            for j = 1, opt.frameRepeat do
                game:advanceAction()
            end
        end

        -- Sleep between episodes:
        sys.sleep(1)
        local score = game:getTotalReward()
        print("Total score: ", score)
    end
    game:close()
end

-- run main program:
main()
