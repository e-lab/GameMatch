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



function torchSetup(_opt)
    _opt = _opt or {}
    local opt = table.copy(_opt)
    assert(opt)

    -- preprocess options:
    --- convert options strings to tables
    if opt.pool_frms then
        opt.pool_frms = str_to_table(opt.pool_frms)
    end
    if opt.env_params then
        opt.env_params = str_to_table(opt.env_params)
    end
    if opt.agent_params then
        opt.agent_params = str_to_table(opt.agent_params)
        opt.agent_params.gpu       = opt.gpu
        opt.agent_params.best      = opt.best
        opt.agent_params.verbose   = opt.verbose
        if opt.network ~= '' then
            opt.agent_params.network = opt.network
        end
    end

    --- general setup
    if not opt.threads then
        opt.threads = 8
    end
    torch.setnumthreads(opt.threads)
    if not opt.verbose then
        opt.verbose = 10
    end
    if opt.verbose >= 1 then
        print('Torch Threads:', torch.getnumthreads())
    end

    --- set gpu device
    if opt.gpu and opt.gpu >= 0 then
        require 'cutorch'
        require 'cunn'
        if opt.gpu == 0 then
            local gpu_id = tonumber(os.getenv('GPU_ID'))
            if gpu_id then opt.gpu = gpu_id+1 end
        end
        if opt.gpu > 0 then cutorch.setDevice(opt.gpu) end
        opt.gpu = cutorch.getDevice()
        print('Using GPU device id:', opt.gpu-1)
    else
        opt.gpu = -1
        if opt.verbose >= 1 then
            print('Using CPU code only. GPU device id:', opt.gpu)
        end
    end

    return opt
end


function setup(_opt)
    assert(_opt)

    --preprocess options:
    --- convert options strings to tables
    _opt.pool_frms = str_to_table(_opt.pool_frms)
    _opt.env_params = str_to_table(_opt.env_params)
    _opt.agent_params = str_to_table(_opt.agent_params)
    if _opt.agent_params.transition_params then
        _opt.agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)
    end

    --- first things first
    local opt = torchSetup(_opt)

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
    _opt.agent_params.verbose = _opt.verbose
    if not _opt.agent_params.state_dim then
        _opt.agent_params.state_dim = gameEnv:nObsFeature()
    end

    -- local agent = dqn[_opt.agent](_opt.agent_params)

    if opt.verbose >= 1 then
        print('Set up Torch using these options:')
        for k, v in pairs(opt) do
            print(k, v)
        end
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

function table.copy(t)
    if t == nil then return nil end
    local nt = {}
    for k, v in pairs(t) do
        if type(v) == 'table' then
            nt[k] = table.copy(v)
        else
            nt[k] = v
        end
    end
    setmetatable(nt, table.copy(getmetatable(t)))
    return nt
end
