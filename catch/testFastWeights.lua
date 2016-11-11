--
-- Eugenio Culurciello
-- November 2016
-- test fast weights module
--

torch.manualSeed(0)

require 'nn'
require 'FastWeights'

local nFW = 3
local nFeat = 4
local a = torch.randn(nFeat)
print(sys.COLORS.blue .. 'input:'); print(a)

local net = nn.FastWeights(nFW, nFeat) -- the fast weight loop memory

-- the fast weight memory buffer is filled:
for i = 1, nFW do net:updatePrevOuts(torch.randn(nFeat)) end


print(sys.COLORS.red .. 'prevOuts before:')
for i = 1, nFW do print(net.prevOuts[i]) end

local b = net:forward(a)
print(sys.COLORS.green .. 'output:'); print(b)

-- add to mem:
net:updatePrevOuts(b)

print(sys.COLORS.red .. 'prevOuts after:')
for i=1,nFW do print(net.prevOuts[i]) end
