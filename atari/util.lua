local ut = torch.class('ut')
function ut:__init()
   self.logger = optim.Logger(paths.concat(opt.savedir,'ms_acc_loss.log'))
   self.logger:setNames{'ms','accuracy', 'loss'}
end
function ut:write(time, acc, loss)
   self.logger:add{time, acc, loss}
end
-- memory for experience replay:
function ut:Memory(maxMemory, discount)
    local memory


    if opt.playFile ~= '' then
        memory = torch.load(opt.playFile)
        print('Loaded experience replay memory with play file:', opt.playFile)
        opt.maxMemory = #memory -- resize this to loaded file
    else
        memory = {}
        print('Initialized empty experience replay memory')
    end

    -- Appends the experience to the memory.
    function memory.remember(memoryInput)
        table.insert(memory, memoryInput)
        if (#memory > opt.maxMemory) then
            -- Remove the earliest memory to allocate new experience to memory.
            table.remove(memory, 1)
        end
    end

    function memory.getBatch(batchSize, nbActions, nbStates, nSeq)
        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch)
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)

        local inputs = torch.zeros(batchSize, nSeq, nbStates)
        local targets = torch.zeros(batchSize, nSeq, nbActions)

        -- create inputs and targets:
        for i = 1, chosenBatchSize do
            local randomIndex = torch.random(1, memoryLength)
            inputs[i] = memory[randomIndex].states:float() -- save as byte, use as float
            targets[i]= memory[randomIndex].actions:float()
        end
        if opt.useGPU then inputs = inputs:cuda() targets = targets:cuda() end

        return inputs, targets
    end

    return memory
end

-- Converts input tensor into table of dimension equal to first dimension of input tensor
-- and adds padding of zeros, which in this case are states
local function tensor2Table(inputTensor, padding, state)
   local outputTable = {}
   for t = 1, inputTensor:size(1) do outputTable[t] = inputTensor[t] end
   for l = 1, padding do outputTable[l + inputTensor:size(1)] = state[l]:clone() end
   return outputTable
end

-- training code:
function ut:trainNetwork(model, state, inputs, targets, criterion, sgdParams, nSeq, nbActions)
    local loss = 0
    local x, gradParameters = model:getParameters()

    local function feval(x_new)
        gradParameters:zero()
        inputs = {inputs, table.unpack(state)} -- attach states
        local out = model:forward(inputs)
        local predictions = torch.Tensor(nSeq, opt.batchSize, nbActions)
        if opt.useGPU then predictions = predictions:cuda() end
        -- create table of outputs:
        for i = 1, nSeq do
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

    local _, fs = optim.sgd(feval, x, sgdParams)

    loss = loss + fs[1]
    return loss
end

function ut:preProcess(im)
  local net = nn.SpatialMaxPooling(4,4,4,4)
  -- local pooled = net:forward(im[1])
  local pooled = net:forward(im[1][{{},{94,194},{9,152}}])
  -- print(pooled:size())
  local out = image.scale(pooled, opt.gridSize, opt.gridSize):sum(1):div(3)
  return out
end
return ut
