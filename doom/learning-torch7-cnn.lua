#!/usr/bin/env th
-- learning-torch7-rnn.lua
-- E. Culurciello, December 2016
-- based on learning tensorflow

local base_path="/Users/eugenioculurciello/Desktop/ViZDoom/"
package.path = package.path .. ";"..base_path.."lua/vizdoom/?.lua"
require "vizdoom.init"
require "nn"
require "torch"
require "sys"
require "image"

local opt = {}
opt.threads = 8
opt.seed = 1

torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)

-- Q-learning settings
local learning_rate = 0.00025
local discount_factor = 0.99
local epochs = 20
local learning_steps_per_epoch = 2000
local replay_memory_size = 10000

-- NN learning settings
local batch_size = 64

-- Training regime
local test_episodes_per_epoch = 100

-- Other parameters
local frame_repeat = 12
local resolution = {30, 45}
local episodes_to_watch = 10

local model_savefile = "results/model.net"
local save_model = true
local load_model = false
local skip_learning = false

 -- Configuration file path
local config_file_path = base_path.."scenarios/simpler_basic.cfg"
-- config_file_path = "../../scenarios/rocket_basic.cfg"
-- config_file_path = "../../scenarios/basic.cfg"

local actions = {
    [1] = torch.IntTensor({1,0,0}),
    [2] = torch.IntTensor({0,1,0}),
    [3] = torch.IntTensor({0,0,1})
}

-- Converts and down-samples the input image
local function preprocess(inImage)
  return image.scale(inImage, unpack(resolution))
end

-- class ReplayMemory:
local memory = {}
local function ReplayMemory(capacity)
    local channels = 1
    memory.s1 = torch.zeros(capacity, resolution[1], resolution[2], channels)
    memory.s2 = torch.zeros(capacity, resolution[1], resolution[2], channels)
    memory.a = torch.zeros(capacity)
    memory.r = torch.zeros(capacity)
    memory.isterminal = torch.zeros(capacity)

    memory.capacity = capacity
    memory.size = 0
    memory.pos = 1

    function memory.addTransition(s1, action, s2, isterminal, reward)
        memory.s1[{memory.pos, {}, {}, 1}] = s1
        memory.a[memory.pos] = action
        if not isterminal then
            memory.s2[{memory.pos, {}, {}, 1}] = s2
        end
        memory.isterminal[memory.pos] = isterminal and 1 or 0
        memory.r[memory.pos] = reward

        memory.pos = (memory.pos + 1) % memory.capacity
        memory.size = math.min(memory.size + 1, memory.capacity)
    end

    function memory.getSample(sample_size)
        -- i = sample(range(0, memory.size), sample_size)
        i = torch.random(1, memory.size)
        return memory.s1[i], memory.a[i], memory.s2[i], memory.isterminal[i], memory.r[i]
    end

end


function createNetwork(available_actions_count)
    -- Create the input variables
    -- Add 2 convolutional layers with ReLu activation
    -- conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
    --                                        activation_fn=tf.nn.relu,
    --                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    --                                        biases_initializer=tf.constant_initializer(0.1))
    -- conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
    --                                        activation_fn=tf.nn.relu,
    --                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
    --                                        biases_initializer=tf.constant_initializer(0.1))
    -- conv2_flat = tf.contrib.layers.flatten(conv2)
    -- fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                            -- weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            -- biases_initializer=tf.constant_initializer(0.1))

    -- q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                          -- weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          -- biases_initializer=tf.constant_initializer(0.1))
    -- best_a = tf.argmax(q, 1)

    -- loss = tf.contrib.losses.mean_squared_error(q, target_q_)
    -- loss = nn.MSECriterion()

    -- optimizer = tf.train.RMSPropOptimizer(learning_rate)
    -- Update the parameters according to the computed gradient using RMSProp.
    -- train_step = optimizer.minimize(loss)

    model = nn.Sequential()
    model:add(nn.SpatialConvolution(1,8,6,6,3,3))
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(8,8,3,3,2,2))
    model:add(nn.ReLU())
    -- model:add(nn.View(8*6*6))
    -- model:add(nn.Linear(8*6*6, 128))
    -- model:add(nn.ReLU())
    -- model:add(nn.Linear(128, available_actions_count))

    function functionLearn(s1, target_q)
        feed_dict = {s1_=s1, target_q_=target_q}
        -- l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l
    end

    function functionGetQValues(state)
        -- return session.run(q, feed_dict={s1_=state})
    end

    function functionGetBestAction(state)
        return 2--session.run(best_a, feed_dict={s1_=state})
    end

    function functionSimpleGetBestAction(state)
        return 2-- return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))[0]
    end

    return functionLearn, functionGetQValues, functionSimpleGetBestAction
end

function learnFromMemory()
    -- Learns from a single transition (making use of replay memory).
    -- s2 is ignored if s2_isterminal

    -- Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size then
        s1, a, s2, isterminal, r = memory.getSample(batch_size)

        q2 = torch.max(getQValues(s2), 1)
        target_q = get_q_values(s1)
        -- target differs from q only for the selected action. The following means:
        -- target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        -- target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        learn(s1, target_q)
    end
end

function performLearningStep(epoch)
    -- Makes an action according to eps-greedy policy, observes the result
    -- (next state, reward) and learns from the transition

    function explorationRate(epoch)
        --  Define exploration rate change over time
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  -- 10% of learning time
        eps_decay_epochs = 0.6 * epochs  -- 60% of learning time

        if epoch < const_eps_epochs then
            return start_eps
        elseif epoch < eps_decay_epochs then
            -- Linear decay
            return start_eps - (epoch - const_eps_epochs) /
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else
            return end_eps
        end
    end

    s1 = preprocess(game:getState().screenBuffer)

    -- With probability eps make a random action:
    eps = explorationRate(epoch)
    if torch.uniform() <= eps then
        a = torch.random(1, #actions)
    else
        -- Choose the best action according to the network:
        a = get_best_action(s1)
    end
    reward = game:makeAction(actions[a], frame_repeat)

    isterminal = game:isEpisodeFinished()
    if not isterminal then s2 = preprocess(game:getState().screenBuffer) else s2 = nil end

    -- Remember the transition that was just experienced:
    memory.addTransition(s1, a, s2, isterminal, reward)

    learnFromMemory()
end

-- Creates and initializes ViZDoom environment.
function initialize_vizdoom(config_file_path)
    print("Initializing doom...")
    game = vizdoom.DoomGame()
    game:setViZDoomPath(base_path.."bin/vizdoom")
    game:setDoomGamePath(base_path.."scenarios/freedoom2.wad")
    game:loadConfig(config_file_path)
    game:setWindowVisible(false)
    game:setMode(vizdoom.Mode.PLAYER)
    game:setScreenFormat(vizdoom.ScreenFormat.GRAY8)
    game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)
    game:init()
    print("Doom initialized.")
    return game
end

function main()
    -- Create Doom instance
    game = initialize_vizdoom(config_file_path)

    -- Action = which buttons are pressed
    n = game:getAvailableButtonsSize()
    -- actions = [list(a) for a in it.product([0, 1], repeat=n)]

    -- Create replay memory which will store the 
    ReplayMemory(replay_memory_size)

    learn, getQValues, getBestAction = createNetwork(#actions)
    
    if load_model then
        print("Loading model from: ", model_savefile)
        saver.restore(session, model_savefile)
    else
        -- init = tf.initialize_all_variables()
        -- session.run(init)
    end
    print("Starting the training!")

    time_start = sys.tic()
    if not skip_learning then
        for epoch = 1, epochs do
            print(string.format("\nEpoch %d\n-------", epoch))
            train_episodes_finished = 0
            train_scores = {}

            print("Training...")
            game:newEpisode()
            for learning_step = 1, learning_steps_per_epoch do
                -- game:makeAction(actions[2]) -- to test
                performLearningStep(epoch)
                if game:isEpisodeFinished() then
                    score = game:getTotalReward()
                    table.insert(train_scores, score)
                    game:newEpisode()
                    train_episodes_finished = train_episodes_finished + 1
                end
            end

            print(string.format("%d training episodes played.", train_episodes_finished))

            train_scores = torch.Tensor(train_scores)

            print(string.format("Results: mean: %.1f+//-%.1f, min: %.1f, max: %.1f", 
                train_scores:mean(), train_scores:std(), train_scores:min(), train_scores:max()))

            print("\nTesting...")
            test_episode = {}
            test_scores = {}
            for test_episode =1, test_episodes_per_epoch do
                game:newEpisode()
                while not game:isEpisodeFinished() do
                    state = preprocess(game:getState().screenBuffer)
                    best_action_index = getBestAction(state)
                    
                    game:makeAction(actions[best_action_index], frame_repeat)
                end
                r = game:getTotalReward()
                table.insert(test_scores, r)
            end

            test_scores = torch.Tensor(test_scores)
            print(string.format("Results: mean: %.1f+//-%.1f, min: %.1f, max: %.1f",
                test_scores:mean(), test_scores:std(), test_scores:min(), test_scores:max()))

            print("Saving the network weigths to:", model_savefile)
            -- torch.save(model_savefile, model)
            
            print(string.format("Total elapsed time: %.2f minutes", sys.toc()/60.0))
        end
    end
    
    game:close()
    print("======================================")
    print("Training finished. It's time to watch!")

    -- Reinitialize the game with window visible
    game:setWindowVisible(true)
    game:setMode(vizdoom.Mode.ASYNC_PLAYER)
    game:init()

    for i = 1, episodes_to_watch do
        game:newEpisode()
        while not game:isEpisodeFinished() do
            state = preprocess(game:getState().screenBuffer)
            best_action_index = getBestAction(state)

            -- Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game:makeAction(actions[best_action_index])
            for j = 1, frame_repeat do
                game:advanceAction()
            end
        end

        -- Sleep between episodes
        sys.sleep(1)
        score = game:getTotalReward()
        print("Total score: ", score)
    end
    game:close()
end

-- run main program:
main()
