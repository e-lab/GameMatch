--
-- Eugenio Culurciello
-- November 2016
-- fast weights: https://arxiv.org/abs/1610.06258
--
-- Usage: computes the A(t) * h_s(t+1) loop
-- after you need to perform layer normalization (LN) and obtain:
--       h_s+1(t + 1) = f(LN[W*h(t) + C*x(t) + A(t)*h_s(t + 1)])
-- after also call: FastWeights:updatePrevOuts( h_s+1(t + 1) )
--

local FastWeights, parent = torch.class('nn.FastWeights', 'nn.Module')

function FastWeights:__init(nFW, nFeat, eta, lambda)
   parent.__init(self)
   self.eta = eta or 0.5
   self.lambda = lambda or 0.95
   self.nFW = nFW
   self.nFeat = nFeat
   self.prevOuts = {} -- here we store the previous nFW outputs to compute updates
   for i = 1, nFW do
      table.insert(self.prevOuts, torch.zeros(self.nFeat))
    end
end

-- function FastWeights:_lazyInit()
--    if not self.prevOuts then
--       self.prevOuts = {} -- here we store the previous nFW outputs to compute updates
--       for i = 1, nFW do
--         table.insert(self.prevOuts, torch.zeros(self.nFeat))
--       end
--     end
-- end

function FastWeights:updatePrevOuts(item)
   -- store current output to prevOuts
   table.insert(self.prevOuts, 1, item) -- add new states to 1st place
   table.remove(self.prevOuts) -- remove oldest states from list
end

function FastWeights:updateOutput(input)
   -- self:_lazyInit()
   -- print(self.prevOuts)
   local hSum = torch.zeros(self.nFeat)

   -- fast weights update:
   for f = 1, self.nFW do -- 1 = most recent, nFW = most past
      local prod = self.prevOuts[f]:view(1,-1) * input -- input = current hidden state
      hSum = hSum + self.prevOuts[f]:clone():mul(prod[1]):mul(self.lambda^(f))
   end
   local hFW = self.eta * hSum

   self.output = hFW -- output is current fast weights output

   return self.output
end

function FastWeights:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

function FastWeights:type(type, tensorCache)
    self._indices = nil
    parent.type(self, type, tensorCache)
    return self
end

function FastWeights:clearState()
   nn.utils.clear(self, '_indices', '_output')
   return parent.clearState(self)
end
