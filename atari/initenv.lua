-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code

-- inspired by: https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner

dqn = {}

require 'torch'
require 'nn'
require 'nngraph'
require 'image'
require 'model'
require 'optim'


function setup(_opt)
    assert(_opt)

    --preprocess options:
    -- convert options strings to tables
    _opt.pool_frms = str_to_table(_opt.pool_frms)
    _opt.env_params = str_to_table(_opt.env_params)

    -- load training framework and environment
    local framework = assert(require(opt.framework))

    local gameEnv = framework.GameEnvironment(opt)
    local gameActions = gameEnv:getActions()

    return gameEnv, gameActions
end


--- other functions
function str_to_table(str)
    if type(str) == 'table' then
        return str
    end
    if not str or type(str) ~= 'string' then
        if type(str) == 'table' then
            return str
        end
        return {}
    end
    local ttr
    if str ~= '' then
        local ttx=tt
        loadstring('tt = {' .. str .. '}')()
        ttr = tt
        tt = ttx
    else
        ttr = {}
    end
    return ttr
end
