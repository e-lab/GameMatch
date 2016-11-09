--------------------------------------------------------------------------------
-- A simple RNN block + fast weights: https://arxiv.org/abs/1610.06258
-- Written by: Abhishek Chaurasia
-- fast weight added by Eugenio Culurciello, November 2016
--------------------------------------------------------------------------------

local RNNfw = {}

-- Fast Weights version of RNN
-- This RNN cell inputs contains a list of the most recent hidden states
-- this list is organized by the main script, not here
-- here the prototype implements the inner loop of fast-weights computation
-- and also layer normalization (fast weight can explode!)

--[[
                    +-----------+
                    |           |
                    |   +----+  |
                    V   |    |--+
              +--->(+)->| h1 |
              |x1       |    |------+
              |         +----+      |
              |                     |
              |                     |
              |                     |
              |     +-----------+   |
              |     |           |   |
      +-----+ |     |   +----+  |   |     +---+
      |   1 | |     V   |    |--+   +---->|   |
      | x:2 +-+--->(+)->| h2 |   +------->| y |
      |   3 | |x2       |    |---+  +---->|   |
      +-----+ |         +----+      |     +---+
              |                     |
              |                     |
              |                     |
              |     +-----------+   |
              |     |           |   |
              |     |   +----+  |   |
              |     V   |    |--+   |
              +--->(+)->| h3 |      |
               x3       |    |------+
                        +----+

--]]

-- n   : # of inputs
-- d   : # of neurons in hidden layer
-- nHL : # of hidden layers
-- K   : # of output neurons
-- nfw : # of previous vectors to save for fast weights
-- eta : fast learning rate (default 0.5)
-- lambda : fast weights decay rate (default 0.95)

-- Returns a simple RNN model
local function getPrototype(n, d, nHL, K, nfw)
  local eta = 0.5
  local lambda = 0.95
  local prevStates = torch.Tensor(nfw,n,d) -- store previous states to be used by fast weights
  local inputs = {}
  table.insert(inputs, nn.Identity()())       -- input X
  for j = 1, nHL do
    table.insert(inputs, nn.Identity()())    -- previous states h[j] + nfw older states for fast weights
    -- 1 = most recent, nfw+1 = most past
  end

  local x, nIn
  local outputs = {}
  for j = 1, nHL do
    if j == 1 then
      x = inputs[j]
         nIn = n
      else
         x = outputs[j-1]
         nIn = d
      end

      local hPrev = inputs[j+1][1] -- [1] is most recent state
      print(hPrev)

      -- fast weight loop:
      local hFW
      for f = 2, nfw+1 do
        hFW = lamba^(f-1) * inputs[j+1][f] * [inputs[j+1][f]:view(1,-1) * hPrev]
      end
      hFW = hFW * eta

      -- layer normalization:
      hFW = hFW:add(-hfW:mean())
      hFW = hFW:div(-hfW:std())
      
      -- Concat input with previous state
      local nextH = ({x, hFW} - nn.JoinTable(1) - nn.Linear(nIn + d, d) - nn.Tanh())
      table.insert(outputs, nextH)
   end

   local logsoft = (outputs[#outputs] - nn.Linear(d, K) - nn.LogSoftMax())
   table.insert(outputs, logsoft)

   -- Output is table with {h, prediction}
   return nn.gModule(inputs, outputs)
end


-- Links all the RNN models, given the # of sequences
function RNN.getModel(n, d, nHL, K, T, nfw)
   local prototype = getPrototype(n, d, nHL, K)

   local clones = {}
   for i = 1, T do
      clones[i] = prototype:clone('weight', 'bias', 'gradWeight', 'gradBias')
   end

   local inputSequence = nn.Identity()()        -- Input sequence
   local H0 = {}                                -- Initial states of hidden layers
   local H = {}                                 -- Intermediate states
   local outputs = {}

   -- Linking initial states to intermediate states
   for l = 1, nHL do
      H0[l] = nn.Identity()()
      H[l] = H0[l]
   end

   local splitInput = inputSequence - nn.SplitTable(1)

   for i = 1, T do
      local x = (splitInput - nn.SelectTable(i))

      local tempStates = ({x, table.unpack(H)} - clones[i])

      outputs[i] = (tempStates - nn.SelectTable(nHL + 1))  -- Prediction

      if i < T then
         for l = 1, nHL do                         -- State values passed to next sequence
            H[l] = (tempStates - nn.SelectTable(l))
         end
      else
         for l = 1, nHL do                         -- State values passed to next sequence
            outputs[T + l] = (tempStates - nn.SelectTable(l))
         end
      end
   end

   -- Output is table of {Predictions, Hidden states of last sequence}
   local g = nn.gModule({inputSequence, table.unpack(H0)}, outputs)

   return g, clones[1]
end

return RNN
