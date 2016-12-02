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

    function memory.getBatch(batchSize, nbActions, nbStates, nSeq, dumy)
        -- We check to see if we have enough memory inputs to make an entire
        -- batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough
        -- experience to fill a batch)
        local memoryLength = #memory
        --local chosenBatchSize = math.min(batchSize, memoryLength)
        local chosenBatchSize = batchSize --math.min(batchSize, memoryLength)
        local inputs = torch.zeros(
                                     batchSize, nSeq, dumy:size(1),
                                     dumy:size(2), dumy:size(3))
        local targets = torch.zeros(batchSize, nSeq, nbActions)

        -- create inputs and targets:
        for i = 1, chosenBatchSize do
            local randomIndex = torch.random(1, memoryLength)
            inputs[i] = memory[randomIndex].states:float()
            targets[i]= memory[randomIndex].actions:float()
        end
        if opt.useGPU then inputs = inputs:cuda() targets = targets:cuda() end

        return inputs, targets
    end
    function memory.getLengty()
       return #memory
    end

    return memory
end
function prepareState(q,predOpt)
 local RNNhProto = {}
 for i =1 , #q do
    if i == 1 then
       -- This is defined in the model code
       RNNhProto[i] = torch.zeros(
          predOpt.channels[predOpt.layers+1],
          predOpt.height/2, predOpt.width/2)
    else
       RNNhProto[i] = q[i]:clone()
    end
 end
 return RNNhProto
end
function shipToGPU(table)
   if type(table) == 'table' then
      for i = 1, #table do
         table[i] = table[i]:cuda()
      end
   else
      table = table:cuda()
      return table
   end
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
function trainNetwork(
   model, state, inputs, targets, criterion, sgdParams, nSeq, nbActions, batch)
    local loss = 0
    local x, gradParameters = model:getParameters()

    local function feval(x_new)
        gradParameters:zero()
        inputs = {inputs, table.unpack(state)} -- attach states
        if opt.useGPU then shipToGPU(inputs) end
        local out = model:forward(inputs)
        local predictions = torch.Tensor(nSeq, batch, nbActions)
        if opt.useGPU then predictions = predictions:cuda() end
        -- create table of outputs:
        for i = 1, nSeq do
            predictions[i] = out[i]
        end
        predictions = predictions:transpose(2,1)
        -- print('in', inputs) print('outs:', out) print('targets',
        -- {targets}) print('predictions', {predictions})
        local loss = criterion:forward(predictions, targets)
        local grOut = criterion:backward(predictions, targets)
        grOut = grOut:transpose(2,1)
        local gradOutput = {}
        for i = 1, grOut:size(1) do
            gradOutput[i] = grOut[i]
        end
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
function getBatchInput(b, seq, height, width, L, channels, mode)
   -- Input for the gModule
   local x = {}

   if mode == 1 then
      x[1] = torch.randn(b, channels[1], height, width)             -- Image
   elseif mode == 2 then
      x[1] = torch.randn(b, seq, channels[1], height, width)        -- Image
   end
   x[3] = torch.zeros(b, channels[1], height, width)                -- C1[0]
   x[4] = torch.zeros(b, channels[1], height, width)                -- H1[0]
   x[5] = torch.zeros(b, 2*channels[1], height, width)              -- E1[0]

   for l = 2, L do
      height = height/2
      width = width/2
      x[3*l]   = torch.zeros(b, channels[l], height, width)         -- C1[0]
      x[3*l+1] = torch.zeros(b, channels[l], height,width)          -- Hl[0]
      x[3*l+2] = torch.zeros(b, 2*channels[l], height, width)       -- El[0]
   end
   height = height/2
   width = width/2
   x[2] = torch.zeros(b, channels[L+1], height, width)              -- RL+1
  local y = {}
  for i = 2, #x do
     y[i-1] = x[i]
   end
   return x[1], y
end

function getInput(seq, height, width, L, channels, mode)
   -- Input for the gModule
   local x = {}

   if mode == 1 then
      x[1] = torch.randn(channels[1], height, width)             -- Image
   elseif mode == 2 then
      x[1] = torch.randn(seq, channels[1], height, width)        -- Image
   end
   x[3] = torch.zeros(channels[1], height, width)                -- C1[0]
   x[4] = torch.zeros(channels[1], height, width)                -- H1[0]
   x[5] = torch.zeros(2*channels[1], height, width)              -- E1[0]

   for l = 2, L do
      height = height/2
      width = width/2
      x[3*l]   = torch.zeros(channels[l], height, width)         -- C1[0]
      x[3*l+1] = torch.zeros(channels[l], height,width)          -- Hl[0]
      x[3*l+2] = torch.zeros(2*channels[l], height, width)       -- El[0]
   end
   height = height/2
   width = width/2
   x[2] = torch.zeros(channels[L+1], height, width)              -- RL+1
  local y = {}
  for i = 2, #x do
     y[i-1] = x[i]
   end

   return x[1], y
end

return ut
