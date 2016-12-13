-- Eugenio Culurciello
-- November 2016
-- Deep RNN for reinforcement online learning

-- memory for experience replay:
local memory = {}
function initMemory(maxMemory, nSeq, nbStates)
    memory.states = torch.zeros(maxMemory, nSeq, nbStates)
    memory.actions = torch.ones(maxMemory, nSeq)
    memory.capacity = maxMemory
    memory.size = 0
    memory.pos = 1
   
    -- batch buffers:
    local binputs = torch.zeros(opt.batchSize, nSeq, nbStates)
    local btargets = torch.ones(opt.batchSize, nSeq)

    if opt.playFile ~= '' then
        memory = torch.load(opt.playFile)
        print('Loaded experience replay memory with play file:', opt.playFile)
    else
        print('Initialized empty experience replay memory')
    end

    -- Appends the experience to the memory.
    function memory.remember(seqs,acts)
      memory.states[memory.pos] = seqs:clone()
      memory.actions[memory.pos] = acts:clone()
       
      memory.pos = (memory.pos + 1) % memory.capacity
      if memory.pos == 0 then memory.pos = 1 end -- to prevent issues!
      memory.size = math.min(memory.size + 1, memory.capacity)
    end

    function memory.getBatch(batchSize)
        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch)
        local chosenBatchSize = math.min(opt.batchSize, memory.size)

        -- create inputs and targets:
        local ri = torch.randperm(memory.size)
        for i = 1, chosenBatchSize do
            binputs[i] = memory.states[ri[i]]
            btargets[i]= memory.actions[ri[i]]
        end
        if opt.useGPU then binputs = binputs:cuda() btargets = btargets:cuda() end

        return binputs, btargets
    end
    return memory
end


local trainer = {} -- trainer object

-- Params for Stochastic Gradient Descent (our optimizer).
trainer.sgdParams = {
    learningRate = opt.learningRate,
}

-- Mean Squared Error for our loss function.
trainer.criterion = nn.ClassNLLCriterion()

trainer.poolnet = nn.SpatialMaxPooling(4,4,4,4)
function trainer.preProcess(inImage)
  local pooled = trainer.poolnet:forward(inImage[1][{{},{94,194},{9,152}}])
  local outImage = image.scale(pooled, opt.gridSize, opt.gridSize):sum(1):div(3)
  return outImage
end


-- training code:
function trainer.trainNetwork(model, state, inputs, targets, nSeq)
    local loss = 0
    local x, gradParameters = model:getParameters()

    local function feval(x_new)
      local loss = 0
      local grOut = {}
      -- print(state)
      -- print(targets)
      -- print(inputs)
      -- io.read()
      local inputs = { inputs, table.unpack(state) } -- attach RNN states to input
      local out = model:forward(inputs)
      -- process each sequence step at a time:
      for i = 1, nSeq do
          gradParameters:zero()
          loss = loss + trainer.criterion:forward(out[i], targets[{{},i}])
          grOut[i] = trainer.criterion:backward(out[i], targets[{{},i}])
      end
      table.insert(grOut, state) -- attach RNN states to grad output
      model:backward(inputs, grOut)
      -- print(loss)
      return loss, gradParameters
  end

    local _, fs = optim.rmsprop(feval, x, trainer.sgdParams)

    loss = loss + fs[1]
    return loss
end

return trainer