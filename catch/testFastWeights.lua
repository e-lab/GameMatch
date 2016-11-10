-- 
-- Eugenio Culurciello
-- November 2016
-- test fast weights module
--

require 'nn'
require 'FastWeights'

local nFW = 3
local nFeat = 4
local a = torch.randn(nFeat)
print('input:', a)

local net = nn.FastWeights(nFW, nFeat) -- the fast weight loop memory

-- the fast weight memory buffer is filled:
for i=1,nFW do net:updatePrevOuts(torch.randn(nFeat)) end


print('\nprevOuts before:')
for i=1,nFW do print(net.prevOuts[i]) end

local b = net:forward(a)
print('output:', b)

-- add to mem:
net:updatePrevOuts(b)

print('\nprevOuts after:')
for i=1,nFW do print(net.prevOuts[i]) end
