local class = require 'class'
local Tr = class('Tr')
function Tr:__init(opt)
	require 'nn'
	for name, value in pairs(opt) do
      self[name] = value
   end
	-- Params for Stochastic Gradient Descent (our optimizer).
	self.optimState = {
		 learningRate = opt.learningRate,
		 learningRateDecay = opt.learningRateDecay,
		 weightDecay = opt.weightDecay,
		 momentum = opt.momentum,
		 dampening = 0,
		 nesterov = true
	}
	-- Create the base model.
	local hiddenSize = opt.hiddenSize
	local nbStates = opt.gridSize * opt.gridSize
	local gridSize = opt.gridSize
	local nbActions = opt.nbActions 
	local model = nn.Sequential()
	if opt.modelType == 'mlp' then
		 model:add(nn.Linear(nbStates, hiddenSize))
		 model:add(nn.ReLU())
		 model:add(nn.Linear(hiddenSize, hiddenSize))
		 model:add(nn.ReLU())
		 model:add(nn.Linear(hiddenSize, nbActions))
		 -- test:
		 print('test model:', model:forward(torch.Tensor(nbStates)))
	elseif opt.modelType == 'cnn' then
		 model:add(nn.View(1, gridSize, gridSize))
		 model:add(nn.SpatialConvolution(1, 32, 4,4, 2,2))
		 model:add(nn.ReLU())
		 model:add(nn.View(32*16))
		 model:add(nn.Linear(32*16, hiddenSize))
		 model:add(nn.ReLU())
		 model:add(nn.Linear(hiddenSize, nbActions))
		 -- test:
		 print('test model:', model:forward(torch.Tensor(gridSize, gridSize)))
	else
		 print('model type not recognized')
	end
	self.model = model
   self.we, self.gr = self.model:getParameters()
	-- Mean Squared Error for our loss function.
	local criterion = nn.MSECriterion()
	if self.useGpu then 
      print('Use Gpu')
		require 'cudnn'
		require 'cutorch'
		cutorch.setDevice(self.gpuId)
		self.model:cuda()
		cudnn.convert(self.model,cudnn)
		self.criterion = criterion:cuda()
		self.we = self.we:cuda()
		self.gr = self.gr:cuda()
   else
      self.criterion = criterion
	end
end

--[[ Runs one gradient update using SGD returning the loss.]] --
function Tr:predict(currentState)
	currentState = self:sendGpu(currentState)
	q = self.model:forward(currentState)
	return q
end
function Tr:sendGpu(inputs)
	if self.useGpu then 
		print('send gpu')
		return inputs:cuda()
	else
		return inputs
	end
end
function Tr:train(inputs, targets)
	 inputs = Tr:sendGpu(inputs)
	 targets = Tr:sendGpu(targets)
    local loss = 0
    local function feval(we)
		  self.model:training()
        self.gr:zero()
        local predictions = self.model:forward(inputs)
        local loss = self.criterion:forward(predictions, targets)
        local gradOutput = self.criterion:backward(predictions, targets)
        self.model:backward(inputs, gradOutput)
        return loss, self.gr
    end

    local _, fs = optim.adam(feval, self.we, self.optimState)
    loss = loss + fs[1]
    return loss
end

return Tr
