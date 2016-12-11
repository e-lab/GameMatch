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
  --discount            (default 0.99)       discount factor in learning
  --epsilon             (default 1)          initial value of ϵ-greedy action selection
  --epsilonMinimumValue (default 0.1)        final value of ϵ-greedy action selection
  
  Training parameters:
  --threads               (default 8)        number of threads used by BLAS routines
  --seed                  (default 1)        initial random seed
  -r,--learningRate       (default 0.00025)  learning rate
  --batchSize             (default 64)       batch size for training
  --maxMemory             (default 1e3)      Experience Replay buffer memory
  --epochs                (default 20)       number of training steps to perform

  -- Q-learning settings:
  --learningStepsEpoch    (default 2000)     Learning steps per epoch
  --clampReward                              clamp reward to -1, 1

  -- Model parameters:
  --nSeq                  (default 30)       lenght of sequences of actions the RNN needs to remember
  --fw                                       Use FastWeights or not
  --nLayers               (default 1)        RNN layers
  --nHidden               (default 128)      RNN hidden size
  --nFW                   (default 8)        number of fast weights previous vectors

  -- Training regime
  --testEpisodesEpoch     (default 100)      test episodes per epoch
  --frameRepeat           (default 12)       repeat frame in test mode
  --episodesWatch         (default 10)       episodes to watch after training
  
  Display and save parameters:
  --display                                  display stuff
  --saveDir          (default './results')   subdirectory to save experiments in
]]

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.saveDir)

-- RNN package:
package.path = '../catch/?.lua;' .. package.path
local rnn = require 'RNN'

-- Other parameters
local resolution = {30, 45} -- Y, X sizes of rescaled state / game screen
local nbStates = resolution[1]*resolution[2] -- size of RNN input vector (game state treated as vector here)

local colors = sys.COLORS

-- Configuration file path
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
  return image.scale(inImage, unpack(resolution))
end

-- class ReplayMemory:
local memory = {}
local function ReplayMemory(capacity)
    local channels = 1
    memory.s = torch.zeros(capacity, opt.nSeq, channels*resolution[1]*resolution[2])
    memory.a = torch.zeros(capacity, opt.nSeq, #actions)

    memory.capacity = capacity
    memory.size = 0
    memory.pos = 1

    -- batch buffers:
    memory.bs = torch.zeros(opt.batchSize, opt.nSeq, channels*resolution[1]*resolution[2])
    memory.ba = torch.zeros(opt.batchSize, opt.nSeq, #actions)
    
    function memory.addTransition(state, action)
        memory.s[memory.pos] = state
        memory.a[memory.pos] = action
       
        memory.pos = (memory.pos + 1) % memory.capacity
        if memory.pos == 0 then memory.pos = 1 end -- to prevent issues!
        memory.size = math.min(memory.size + 1, memory.capacity)
    end

    function memory.getSample(sampleSize)
        for i=1,sampleSize do
            local ri = torch.random(1, memory.size)
            memory.bs[i] = memory.s[ri]
            memory.ba[i] = memory.a[ri]
        end
        return memory.bs, memory.ba
    end

end

local sgdParams = {
    learningRate = opt.learningRate,
}

local model, criterion, prototype
local RNNh0Batch = {} -- initial state for training batches
local RNNh0Proto = {} -- initial state - prototype
local RNNhProto = {} -- state to loop through prototype in inference
function createNetwork(available_actions_count)
    -- Create the base RNN model:
    -- here the state is not an image, but a vectorized version of the image
    -- next steps are convRNN models
    if opt.fw then
      print('Created fast-weights RNN with:\n- input size:', nbStates, '\n- number hidden:', opt.nHidden, 
        '\n- layers:', opt.nLayers, '\n- output size:', #actions, '\n- sequence length:', opt.nSeq, 
        '\n- fast weights states:', opt.nFW)
      model, prototype = rnn.getModel(nbStates, opt.nHidden, opt.nLayers, #actions, opt.nSeq, opt.nFW)
    else
      print('Created RNN with:\n- input size:', nbStates, '\n- number hidden:', opt.nHidden, 
          '\n- layers:', opt.nLayers, '\n- output size:', #actions, '\n- sequence lenght:',  opt.nSeq)
      model, prototype = rnn.getModel(nbStates, opt.nHidden, opt.nLayers, #actions, opt.nSeq)
    end

    -- Default RNN initial state set to zero:
    for l = 1, opt.nLayers do
       RNNh0Batch[l] = torch.zeros(opt.batchSize, opt.nHidden)
       RNNh0Proto[l] = torch.zeros(1, opt.nHidden) -- prototype forward does not work on batches
       -- if opt.useGPU then RNNh0Batch[l]=RNNh0Batch[l]:cuda() RNNh0Proto[l]=RNNh0Proto[l]:cuda() end
    end

    -- test model:
    -- print('Testing model and prototype RNN:')
    -- local ttest 
    -- if opt.useGPU then ttest = {torch.CudaTensor(1, nbStates), torch.CudaTensor(1, opt.nHidden)}
    -- else ttest = {torch.Tensor(1, nbStates), torch.Tensor(1, opt.nHidden)} end
    -- print(ttest)
    -- local a = prototype:forward(ttest)
    -- print('TEST prototype:', a)
    -- if opt.useGPU then ttest = {torch.CudaTensor(opt.batchSize, nSeq, nbStates), torch.CudaTensor(opt.batchSize, opt.nHidden)}
    -- else ttest = {torch.Tensor(opt.batchSize, opt.nSeq, nbStates), torch.Tensor(opt.batchSize, opt.nHidden)} end
    -- print(ttest)
    -- local a = model:forward(ttest)
    -- print('TEST model:', a)

    criterion = nn.MSECriterion()

    -- Converts input tensor into table of dimension equal to first dimension of input tensor
    -- and adds padding of zeros, which in this case are states
    local function tensor2Table(inputTensor, padding, state)
       local outputTable = {}
       for t = 1, inputTensor:size(1) do outputTable[t] = inputTensor[t] end
       for l = 1, padding do outputTable[l + inputTensor:size(1)] = state[l]:clone() end
       return outputTable
    end

    -- training code:
    local function functionLearn(seqs, targets, state)
        local params, gradParameters = model:getParameters()

        local function feval(x_new)
            gradParameters:zero()
            local inputs = { seqs, table.unpack(state) } -- attach RNN states to input
            local out = model:forward(inputs)
            local predictions = torch.Tensor(opt.nSeq, opt.batchSize, #actions)
            -- if opt.useGPU then predictions = predictions:cuda() end
            -- create table of outputs:
            for i = 1, opt.nSeq do
                predictions[i] = out[i]
            end
            predictions = predictions:transpose(2,1)
            -- print('in', inputs) print('outs:', out) print('targets', {targets}) print('predictions', {predictions})
            local loss = criterion:forward(predictions, targets)
            local grOut = criterion:backward(predictions, targets)
            grOut = grOut:transpose(2,1)
            local gradOutput = tensor2Table(grOut, 1, state)
            model:backward(inputs, gradOutput)
            return loss, gradParameters
        end

        local _, fs = optim.rmsprop(feval, params, sgdParams)

        return fs[1] -- loss
    end

    -- this is for batch learning. We use the full RNN model forward:
    function functionGetQValues(state)
        return model:forward( {state:view(opt.batchSize, opt.nSeq, nbStates), table.unpack(RNNh0Batch)} )
    end

    -- this is for online learning (single inference, batch = 1)
    -- We use the prototype single cell RNN forward
    function functionGetBestAction(state)
        local q = prototype:forward( {state:view(1, nbStates), table.unpack(RNNh0Proto)} )
        -- Find the max index (the chosen action).
        local max, index = torch.max(q[2][1], 1) -- [2] is the output, [1] is RNN state
        local action = index[1]
        
        return action, q
    end

    return functionLearn, functionGetQValues, functionGetBestAction
end

-- Learns from a single transition (making use of replay memory):
function learnFromMemory()
    -- Get a random minibatch from the replay memory and learns from it:
    if memory.size > opt.batchSize then
        local s, a = memory.getSample(opt.batchSize)
        -- BELOW: RNNh0Batch -- always reset to initial state -- CHANGE LATER! TODO!
        learn(s, a, RNNh0Batch) -- states and actions are inputs and targets to learn
    end
end

function performLearningStep(epoch)
    -- Makes an action according to eps-greedy policy, observes the result
    -- (next state, reward) and learns from the transition
    local steps = 0 -- counts steps to game win
    local sSeq = torch.zeros(opt.nSeq, nbStates) -- store sequence of states in successful run
    local aSeq = torch.zeros(opt.nSeq, #actions) -- store sequence of actions in successful run

    local function explorationRate(epoch)
        --  Define exploration rate change over time:
        local start_eps = opt.epsilon
        local end_eps = opt.epsilonMinimumValue
        local const_eps_epochs = 0.1 * opt.epochs  -- 10% of learning time
        local eps_decay_epochs = 0.6 * opt.epochs  -- 60% of learning time

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

    local s = preprocess(game:getState().screenBuffer):float():div(255)

    -- With probability eps make a random action:
    local eps = explorationRate(epoch)
    local a
    if torch.uniform() <= eps then
        a = torch.random(1, #actions)
    else
        -- Choose the best action according to the network:
        a = getBestAction(s)
    end
    local reward = game:makeAction(actions[a], opt.frameRepeat)

    -- save sequences:
    steps = steps+1
    sSeq[steps] = s:clone()
    aSeq[steps][a] = 1 

    -- if it is a successful sequence, record it and the learn
    if reward > 0 then 
        -- Remember the transition that was just experienced:
        memory.addTransition(sSeq, aSeq)
        -- reset step counter and sequence buffers:
        steps = 0
        sSeq:zero()
        aSeq:zero()
        -- learning step:
        learnFromMemory()
    end

    return eps
end

-- Creates and initializes ViZDoom environment:
function initializeViZdoom(config_file_path)
    print("Initializing doom...")
    game = vizdoom.DoomGame()
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

function main()
    -- Create Doom instance:
    local game = initializeViZdoom(config_file_path)

    -- Action = which buttons are pressed:
    local n = game:getAvailableButtonsSize()
    
    -- Create replay memory which will store the play data:
    ReplayMemory(opt.maxMemory)

    learn, getQValues, getBestAction = createNetwork(#actions) -- note: global functions!
    
    print("Starting the training!")

    local time_start = sys.tic()
    if not skip_learning then
        local epsilon
        for epoch = 1, opt.epochs do
            print(string.format(colors.green.."\nEpoch %d\n-------", epoch))
            local train_episodes_finished = 0
            local train_scores = {}

            print(colors.red.."Training...")
            game:newEpisode()
            for learning_step=1, opt.learningStepsEpoch do
                xlua.progress(learning_step, opt.learningStepsEpoch)
                epsilon = performLearningStep(epoch)
                if game:isEpisodeFinished() then
                    local score = game:getTotalReward()
                    table.insert(train_scores, score)
                    game:newEpisode()
                    train_episodes_finished = train_episodes_finished + 1
                end
            end

            print(string.format("%d training episodes played.", train_episodes_finished))

            train_scores = torch.Tensor(train_scores)

            print(string.format("Results: mean: %.1f, std: %.1f, min: %.1f, max: %.1f", 
                train_scores:mean(), train_scores:std(), train_scores:min(), train_scores:max()))
            -- print('Epsilon value', epsilon)

            print(colors.red.."\nTesting...")
            local test_episode = {}
            local test_scores = {}
            for test_episode=1, opt.testEpisodesEpoch do
                xlua.progress(test_episode, opt.testEpisodesEpoch)
                game:newEpisode()
                while not game:isEpisodeFinished() do
                    local state = preprocess(game:getState().screenBuffer:float():div(255))
                    local best_action_index = getBestAction(state)
                    
                    game:makeAction(actions[best_action_index], opt.frameRepeat)
                end
                local r = game:getTotalReward()
                table.insert(test_scores, r)
            end

            test_scores = torch.Tensor(test_scores)
            print(string.format("Results: mean: %.1f, std: %.1f, min: %.1f, max: %.1f",
                test_scores:mean(), test_scores:std(), test_scores:min(), test_scores:max()))

            print("Saving the network weigths to:", opt.saveDir)
            torch.save(opt.saveDir..'/model-'..epoch..'.net', model:float():clearState())
            
            print(string.format(colors.cyan.."Total elapsed time: %.2f minutes", sys.toc()/60.0))
        end
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
        while not game:isEpisodeFinished() do
            local state = preprocess(game:getState().screenBuffer:float():div(255))
            local best_action_index = getBestAction(state)

            game:makeAction(actions[best_action_index])
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
