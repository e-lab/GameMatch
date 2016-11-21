--if opt.fw then require 'FastWeights' end
require 'nngraph'
require 'convRNN'
require 'nn'
nngraph.setDebug(true)
local model = {}
function model:getModel(ich, och, nHL, K, seq, nFW, w, h, action)
   local cr = convRNN(ich, och, 3, 3, 1, 1, nHL)

   local clones = {}
   for i = 1, seq do
      clones[i] = cr:clone('weight', 'bias', 'gradWeight', 'gradBias')
   end

   local input = nn.Identity()()
   local H0 = {}
   local H = {}
   local outputs = {}

   for l = 1, nHL do
      table.insert(H0, nn.Identity()())
      table.insert(H, H0[l])
   end

   local splitInput = input - nn.SplitTable(1)

   for i = 1, seq do
      local x = splitInput - nn.SelectTable(i,i)

      local tmpSta = {x, table.unpack(H)} - clones[i]


      if i < seq then
         if nHL ~= 1 then
            for l = 1, nHL do
                  H[l] = tmpSta - nn.SelectTable(l,l)
            end
         else
            H[1] = tmpSta
         end
      end
      --Extract output per seq
      if nHL ~= 1 then
         outputs[i] = tmpSta - nn.SelectTable(nHL , nHL ) - nn.View(och*w*h) - nn.Linear(och*w*h,action)
      else
         outputs[i] = tmpSta - nn.Identity() - nn.View(och*w*h) - nn.Linear(och*w*h,action)
      end

   end

   local g = nn.gModule({input, table.unpack(H0)}, outputs)

   return g, clones[1]
end

return model
