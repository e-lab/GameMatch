--------------------------------------------------------------------------------
-- A simple RNN block + fast weights: https://arxiv.org/abs/1610.06258
-- Written by: Abhishek Chaurasia
-- fast weight added by Eugenio Culurciello, November 2016
--------------------------------------------------------------------------------

require 'FastWeights'
require 'nngraph' -- IMPORTANT!!! require nngraph after adding our nn module!!!!
-- otherwise it will not inherit the right overloaded functions!

-- nngraph.setDebug(true)

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
-- nFW : # of previous vectors to save for fast weights
-- eta : fast learning rate (default 0.5)
-- lambda : fast weights decay rate (default 0.95)

-- Returns a simple RNN model
local function getPrototype(n, d, nHL, K, nFW)
  -- local eta = 0.5
  -- local lambda = 0.95
  local inputs = {}
  table.insert(inputs, nn.Identity()())       -- input X
  for j = 1, nHL do
    table.insert(inputs, nn.Identity()())  -- nfw older states for fast weights, 1 = most recent, nfw = most past
  end

  local x, nIn
  local outputs = {}
  for j = 1, nHL do
    if j == 1 then
      x = inputs[j]
      x:annotate{name = 'x[t]',
                 graphAttributes = {
                 style = 'filled',
                 fillcolor = 'moccasin'}}
      nIn = n
    else
      x = outputs[j-1]
      nIn = d
    end
      
    local hPrev = inputs[j+1]:annotate{name = 'h^('..j..')[t-1]',
                                graphAttributes = {
                                style = 'filled',
                                fillcolor = 'lightpink'}}

    -- compute fast weights output:
    -- local FWMod = nn.FastWeights(nFW, d)
    
    -- local hFW = {hPrev} - FWMod

    -- next State:
    -- local Wh = {hFW} - nn.Linear(d, d) - nn.Tanh()
    local Wh = {hPrev} - nn.Linear(d, d) - nn.Tanh()
    local Cx = {x} - nn.Linear(n, d) - nn.Tanh()
    -- local nextH = {x, hPrev} - nn.JoinTable(1) - nn.Linear(nIn + d, d) - nn.Tanh()
    local nextH = {Wh, Cx} - nn.CAddTable()
    -- local nextH = {hPrev, Cx, Wh} - nn.CAddTable() - nn.SoftMax()
    -- local nextH = {x, hPrev} - nn.JoinTable(1) - nn.Linear(nIn + d, d) - nn.Tanh() -- original RNN version
    nextH:annotate{name = 'h^('..j..')[t]',
                   graphAttributes = {
                   style = 'filled',
                   fillcolor = 'skyblue'}}

    -- -- add to FastWeight module memory:
    -- if torch.isTensor(nextH) then FWMod:updatePrevOuts(nextH) end
    -- FWMod:updatePrevOuts(nextH)

    table.insert(outputs, nextH)
  end

  local logsoft = (outputs[#outputs] - nn.Linear(d, K) - nn.LogSoftMax())
  logsoft:annotate{name = 'y\'[t]',
                   graphAttributes = {
                   style = 'filled',
                   fillcolor = 'seagreen1'}}
  table.insert(outputs, logsoft)

  local model = nn.gModule(inputs, outputs)

  -- Output is table with {h, prediction}
  return model
end

-- Links all the RNN models, given the # of sequences
function RNNfw.getModel(n, d, nHL, K, T, nFW)
   local prototype = getPrototype(n, d, nHL, K, nFW)

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
             :annotate{name = 'h^('..l..')[0]',
              graphAttributes = {
              style = 'filled',
              fillcolor = 'lightpink'}}
   end

   local splitInput = inputSequence - nn.SplitTable(1)

   for i = 1, T do
      local x = (splitInput - nn.SelectTable(i))
                :annotate{name = 'x['..i..']',
                 graphAttributes = {
                 style = 'filled',
                 fillcolor = 'moccasin'}}

      local tempStates = ({x, table.unpack(H)} - clones[i])
                         :annotate{name = 'RNN['..i..']',
                          graphAttributes = {
                          style = 'filled',
                          fillcolor = 'skyblue'}}

      outputs[i] = (tempStates - nn.SelectTable(nHL + 1))  -- Prediction
                   :annotate{name = 'y\'['..i..']',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'seagreen1'}}

      if i < T then
         for l = 1, nHL do                         -- State values passed to next sequence
            H[l] = (tempStates - nn.SelectTable(l))
                   :annotate{name = 'h^('..l..')['..i..']',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'lightpink'}}
         end
      else
         for l = 1, nHL do                         -- State values passed to next sequence
            outputs[T + l] = (tempStates - nn.SelectTable(l))
                               :annotate{name = 'h^('..l..')['..i..']',
                                graphAttributes = {
                                style = 'filled',
                                fillcolor = 'lightpink'}}
         end
      end
   end

   -- Output is table of {Predictions, Hidden states of last sequence}
   local g = nn.gModule({inputSequence, table.unpack(H0)}, outputs)

   return g, clones[1]
end

return RNNfw