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

local FastWeights, parent = torch.class('nn.FastWeights', 'nn.Identity') -- Identity parent class

--function FastWeights:__init(nFW, batch, ch, w, h, eta, lambda)
function FastWeights:__init(nFW, d, batch, w, h, eta, lambda)
   parent.__init(self)
   self.eta = eta or 0.5
   self.lambda = lambda or 0.95
   self.nFW = nFW
   self.batch = batch
   self.ch = d
   self.w = w
   self.h = h
   self.prevOuts = {} -- here we store the previous nFW outputs to compute updates
   for i = 1, nFW do
      table.insert(self.prevOuts, torch.zeros(self.batch, self.ch, self.w, self.h))
    end
end

function FastWeights:updateOutput(input)
   local hSum = torch.zeros(self.batch, self.ch, self.w, self.h)
   -- fast weights update:
   for f = 1, self.nFW do -- 1 = most recent, nFW = most past
      local prod = self.prevOuts[f]:cmul(input)
      -- Here previous 2D format was prod[1] replace with prod need to take a look later
      hSum = hSum + self.prevOuts[f]:clone():cmul(prod):mul(self.lambda^(f))
   end
   local nextH = self.eta * hSum

   -- store new state in buffer:
   table.insert(self.prevOuts, 1, nextH) -- add new states to 1st place
   table.remove(self.prevOuts) -- remove oldest states from list

   self.output = nextH -- output is current fast weights output
   return self.output
end
