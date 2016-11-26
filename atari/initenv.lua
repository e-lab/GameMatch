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


function gameEnvSetup(_opt)
    assert(_opt)

    --preprocess options:
    -- convert options strings to tables
    _opt.pool_frms = str_to_table(_opt.pool_frms)
    _opt.env_params = str_to_table(_opt.env_params)
    _opt.agent_params = str_to_table(_opt.agent_params)
    if _opt.agent_params.transition_params then
        _opt.agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)
    end

    -- load training framework and environment
    local framework = assert(require(opt.framework))

    local gameEnv = framework.GameEnvironment(opt)
    local gameActions = gameEnv:getActions()

    -- agent options
    _opt.agent_params.actions   = gameActions
    _opt.agent_params.gpu       = _opt.gpu
    _opt.agent_params.best      = _opt.best
    if _opt.network ~= '' then
        _opt.agent_params.network = _opt.network
    end
    _opt.agent_params.verbose = opt.verbose
    if not _opt.agent_params.state_dim then
        _opt.agent_params.state_dim = gameEnv:nObsFeature()
    end

    return gameEnv, gameActions, agent, opt
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