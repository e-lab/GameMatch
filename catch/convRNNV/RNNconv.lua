--------------------------------------------------------------------------------
-- A simple RNN block + fast weights: https://arxiv.org/abs/1610.06258
-- Written by: Abhishek Chaurasia
-- fast weight added by Eugenio Culurciello, November 2016
--------------------------------------------------------------------------------
if opt.fw then require 'FastWeights' end
require 'nngraph' -- IMPORTANT!!! require nngraph after adding our nn module!!!!
-- otherwise it will not inherit the right overloaded functions!
--nngraph.setDebug(true) -- if you need to debug uncomment this
local backend = nn
local scNB = backend.SpatialConvolution:noBias()

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

-- NOTE 2: this fast weights implementation differs from: https://arxiv.org/abs/1610.06258
-- here we use SoftMax to normalize fast weight hidden vector, also FastWeight module saves its outputs to a memory
-- instead of saving the hidden states after being summed to input projection
-- this implementation is similar to a focused attentional mechanisms on recent hidden weights, using a fast associative memory

-- Returns a simple RNN model
local function getConvRNN(n, d, nHL, K, nFW, batch, w, h)
   local inputs = {}
   table.insert(inputs, nn.Identity()())       -- input X
   for j = 1, nHL do
      table.insert(inputs, nn.Identity()())    -- previous states h[j]
   end

   local x, nIn, nextH, hPrev, Wh, Cx, hFW, hFWNormed, logsoft
   local kw, kh, stw, sth, paw, pah = 3, 3, 1, 1, 1, 1
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

      if j == 1 then
        i2h = {x} - scNB(n, d, kw, kh, stw, sth, paw, pah) - nn.ReLU() -- project input to same dimensions as hidden layers
      else
        i2h = {x} - scNB(d, d, kw, kh, stw, sth, paw, pah) - nn.ReLU() -- project input to same dimensions as hidden layers
      end

      if opt.fw then -- compute fast weights output:
        hFW = {hPrev} - nn.FastWeights(nFW, d, batch, w, h)
        hFWNormed = {hFW} - nn.SoftMax() -- SoftMax used to normalize the results
        h2h = hFWNormed - scNB(d, d, kw, kh, stw, sth, paw, pah) - nn.ReLU() -- project fast weights into hidden layer
        nextH = {i2h, h2h} - nn.CAddTable(1,1)
      else
        h2h = hPrev - scNB(d, d, kw, kh, stw, sth, paw, pah) - nn.ReLU()
        nextH = {i2h, h2h} - nn.CAddTable(1,1)
      end
      nextH:annotate{name = 'h^('..j..')[t]',
                     graphAttributes = {
                     style = 'filled',
                     fillcolor = 'skyblue'}}

      table.insert(outputs, nextH)
   end

   local project = outputs[#outputs] - scNB(d, K, kw, kh, stw, sth, paw, pah)
   local action = project - nn.View(K*h*w) - nn.Linear(K*h*w, K) - nn.LogSoftMax()
          action:annotate{name = 'y\'[t]',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'seagreen1'}}
   table.insert(outputs, action)

   -- Output is table with {h, prediction}
   local prototype = nn.gModule(inputs, outputs)
   -- graph.dot(prototype.fg, 'proto', 'proto') -- plot to debug

   return prototype
end

-- Links all the RNN models, given the # of sequences
function RNN.getModel(n, d, nHL, K, T, nFW, batch, w, h)
   local convRNN = getConvRNN(n, d, nHL, K, nFW, batch, w, h)

   local clones = {}
   for i = 1, T do
      clones[i] = convRNN:clone('weight', 'bias', 'gradWeight', 'gradBias')
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

   local splitInput = inputSequence - nn.SplitTable(2)

   for i = 1, T do
      local x = (splitInput - nn.SelectTable(i,i))
                :annotate{name = 'x['..i..']',
                 graphAttributes = {
                 style = 'filled',
                 fillcolor = 'moccasin'}}

      local tempStates = ({x, table.unpack(H)} - clones[i])
                         :annotate{name = 'RNN['..i..']',
                          graphAttributes = {
                          style = 'filled',
                          fillcolor = 'skyblue'}}

      outputs[i] = (tempStates - nn.SelectTable(nHL + 1, nHL + 1))  -- Prediction
                   :annotate{name = 'y\'['..i..']',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'seagreen1'}}

      if i < T then
         for l = 1, nHL do                         -- State values passed to next sequence
            H[l] = (tempStates - nn.SelectTable(l,l))
                   :annotate{name = 'h^('..l..')['..i..']',
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = 'lightpink'}}
         end
      else
         for l = 1, nHL do                         -- State values passed to next sequence
            outputs[T + l] = (tempStates - nn.SelectTable(l,l))
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
