--------------------------------------------------------------------------------
-- A simple RNN block + fast weights: https://arxiv.org/abs/1610.06258
-- Written by: Abhishek Chaurasia
-- fast weight added by Eugenio Culurciello, November 2016
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

-- NOTE: this RNN is more complex than: https://github.com/e-lab/torch7-demos/blob/master/RNN-train-sample/RNN.lua
-- it uses projections of input and hidden space and sums them

-- Returns a simple RNN model
local function getPrototype(n, d, nHL, K, nFW)
   local inputs = {}
   table.insert(inputs, nn.Identity()())       -- input X
   for j = 1, nHL do
      table.insert(inputs, nn.Identity()())    -- previous states h[j]
   end

   local x, nIn, nextH, hPrev, Wh, Cx, hFW, hFWNormed, logsoft
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

      Cx = {x} - nn.Linear(n, d) - nn.ReLU() -- project input to same dimensions as hidden layers
      
      if opt.fw then -- compute fast weights output:
        hFW = {hPrev} - nn.FastWeights(nFW, d)
        hFWNormed = {hFW} - nn.SoftMax() -- SoftMax used to normalize the results
        Wh = {hFWNormed} - nn.Linear(d, d) - nn.ReLU() -- project fast weights into hidden layer
        nextH = {Wh, Cx} - nn.CAddTable()
      else
        Wh = {hPrev} - nn.Linear(d, d) - nn.ReLU()
        nextH = {Wh, Cx} - nn.CAddTable()
      end
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
   local prototype = nn.gModule(inputs, outputs)
   -- graph.dot(prototype.fg, 'proto', 'proto') -- plot to debug

   return prototype
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
   -- graph.dot(g.fg, 'model', 'model') -- plot to debug

   return g, clones[1]
end

return RNN
