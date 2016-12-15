--
-- Eugenio Culurciello
-- November 2016
-- fast weights: https://arxiv.org/abs/1610.06258
--
-- Usage: computes the A(t) * h_s(t+1) loop
-- after you need to perform layer normalization (LN) and obtain:
--       h_s+1(t + 1) = f(LN[W*h(t) + C*x(t) + A(t)*h_s(t + 1)])

local FastWeights, parent = torch.class('nn.FastWeights', 'nn.Identity') -- Identity parent class 

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

function FastWeights:updateOutput(input)
   local hSum = torch.zeros(self.nFeat)
   -- fast weights update:
   for f = 1, self.nFW do -- 1 = most recent, nFW = most past
      -- print(self.prevOuts, self.prevOuts[1], input)
      local prod = input:clone():cmul(self.prevOuts[f]):sum()
      -- print({self.prevOuts[f], input, prod}) --io.read()
      -- local prod = self.prevOuts[f]:view(1,-1) * input
      hSum = hSum + self.prevOuts[f]:clone():mul(prod):mul(self.lambda^(f))
   end
   local nextH = hSum:mul(self.eta):view(1,-1)
   -- print(nextH)

   -- store new state in buffer:
   table.insert(self.prevOuts, 1, nextH) -- add new states to 1st place
   table.remove(self.prevOuts) -- remove oldest states from list

   self.output = nextH -- output is current fast weights output
   return self.output
end