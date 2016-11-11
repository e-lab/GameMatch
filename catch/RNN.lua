--------------------------------------------------------------------------------
-- A simple RNN block
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

if opt.fw then require 'FastWeights' end
require 'nngraph' -- IMPORTANT!!! require nngraph after adding our nn module!!!!
-- otherwise it will not inherit the right overloaded functions!

local RNN = {}

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

-- Returns a simple RNN model
local function getPrototype(n, d, nHL, K, nFW)
   local inputs = {}
   table.insert(inputs, nn.Identity()())       -- input X
   for j = 1, nHL do
      table.insert(inputs, nn.Identity()())    -- previous states h[j]
   end

   local x, nIn, nextH, hPrev, Wh, Cx, hFW, logsoft
   local FWMod -- module for fast weights
   local outputs = {}
   for j = 1, nHL do
      if j == 1 then
         x = inputs[j]:annotate{name = 'x[t]',
                       graphAttributes = {
                       style = 'filled',
                       fillcolor = 'moccasin'}}
         nIn = n
      else
         x = outputs[j-1]
         nIn = d
      end

      hPrev = inputs[j+1]:annotate{name = 'h^('..j..')[t-1]',
                                   graphAttributes = {
                                   style = 'filled',
                                   fillcolor = 'lightpink'}}

      Wh = {hPrev} - nn.Linear(d, d) - nn.Tanh()
      Cx = {x} - nn.Linear(n, d) - nn.Tanh()
      
      if opt.fw then -- compute fast weights output:
        FWMod = nn.FastWeights(nFW, d)
        hFW = {hPrev} - FWMod
        nextH = {Wh, Cx, hFW} - nn.CAddTable()
        if torch.isTensor(nextH) then print('UPDATE!') FWMod:updatePrevOuts(nextH) end -- DOES NOT WORK!!!!!
      else
        nextH = {Wh, Cx} - nn.CAddTable()
      end
      -- local nextH = {x, hPrev} - nn.JoinTable(1) - nn.Linear(nIn + d, d) - nn.Tanh() -- older simple RNN version
      nextH:annotate{name = 'h^('..j..')[t]',
                     graphAttributes = {
                     style = 'filled',
                     fillcolor = 'skyblue'}}

      table.insert(outputs, nextH)
   end

   logsoft = (outputs[#outputs] - nn.Linear(d, K) - nn.LogSoftMax())
          :annotate{name = 'y\'[t]',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'seagreen1'}}
   table.insert(outputs, logsoft)

   -- Output is table with {h, prediction}
   return nn.gModule(inputs, outputs)
end

-- Links all the RNN models, given the # of sequences
function RNN.getModel(n, d, nHL, K, T, nFW)
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

return RNN
