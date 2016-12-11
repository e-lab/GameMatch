-- Eugenio Culurciello
-- November 2016
-- Deep RNN for reinforcement online learning

-- memory for experience replay:
function Memory(maxMemory, discount)
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

    function memory.getBatch(batchSize, nSeq, nbActions, nbStates)
        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch)
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)

        local inputs = torch.zeros(batchSize, nSeq, nbStates)
        local targets = torch.zeros(batchSize, nSeq, nbActions)

        -- create inputs and targets:
        for i = 1, chosenBatchSize do
            local randomIndex = torch.random(1, memoryLength)
            inputs[i] = memory[randomIndex].states
            targets[i] = memory[randomIndex].actions
        end
        if opt.useGPU then inputs = inputs:cuda() targets = targets:cuda() end

        return inputs, targets
    end

    return memory
end


local trainer = {} -- trainer object

-- Params for Stochastic Gradient Descent (our optimizer).
trainer.sgdParams = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}

-- Mean Squared Error for our loss function.
trainer.criterion = nn.MSECriterion()

trainer.poolnet = nn.SpatialMaxPooling(4,4,4,4)
function trainer.preProcess(inImage)
  local pooled = trainer.poolnet:forward(inImage[1][{{},{94,194},{9,152}}])
  local outImage = image.scale(pooled, opt.gridSize, opt.gridSize):sum(1):div(3)
  return outImage
end

-- Converts input tensor into table of dimension equal to first dimension of input tensor
-- and adds padding of zeros, which in this case are states
function trainer.tensor2Table(inputTensor, padding, state)
   local outputTable = {}
   for t = 1, inputTensor:size(1) do outputTable[t] = inputTensor[t] end
   for l = 1, padding do outputTable[l + inputTensor:size(1)] = state[l]:clone() end
   return outputTable
end

-- training code:
function trainer.trainNetwork(model, state, inputs, targets, nSeq, nbActions)
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
        local loss = trainer.criterion:forward(predictions, targets)
        local grOut = trainer.criterion:backward(predictions, targets)
        grOut = grOut:transpose(2,1)
        local gradOutput = trainer.tensor2Table(grOut, 1, state)
        model:backward(inputs, gradOutput)
        return loss, gradParameters
    end

    local _, fs = optim.rmsprop(feval, x, trainer.sgdParams)
    
    loss = loss + fs[1]
    return loss
end

return trainer