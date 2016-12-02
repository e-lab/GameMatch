--------------------------------------------------------------------------------
-- This script returns prednet model.
-- To get the model, first call __init function with 'opt' as parameter
-- Then call getModel(), which will return the prototype
-- as well as the whole model
--
-- opt is a table containing following indexes:
-- channels, datapath, dev, display, height, layers, learningRate,
-- learningRateDecay, lstmLayer, momentum, nGPU, saveGraph, seq,
-- weightDecay, width
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

local prednet = {}

require 'nngraph'

--nngraph.setDebug(true)
local sf = string.format

-- included local packages
local convLSTM = paths.dofile('convLSTM.lua')

function prednet:__init(opt)
   -- Input/Output channels for A of every layer
   self.channels = opt.channels
   self.layers = opt.layers
   self.seq = opt.seq
   self.height = opt.height
   self.width  = opt.width
   self.saveGraph = opt.saveGraph or false
   self.dev = opt.dev or 'cpu'
   self.lstmLayer = opt.lstmLayer or 1
   self.K = opt.K

   if self.saveGraph then paths.mkdir('graphs') end
end

-- Macros
local SC = nn.SpatialConvolution
local gaA  = {color = 'blue', fontcolor = 'blue'}
local gaAh = {style = 'filled', fillcolor = 'skyblue'}
local gaE  = {style = 'filled', fillcolor = 'lightpink'}
local gaR  = {style = 'filled', fillcolor = 'springgreen'}

--[[
    E: Error
    C: Cell State
    R: Hidden State (H)

                 Rl+1[t]            Rl[t-1]   El[t-1]
                   |                   |         |
                   |                   |         |
                   V                   V         V
            +--------------+      +-------------------+
            |   Upsample   |      | Rl[t-1] | El[t-1] |
            +--------------+      +-------------------+
                   |                        |
                   |                        |
                   V                        |
               +-------+                    |
               |       |                    |
    Cl[t-1]--->| Rl[t] |<-------------------+
               |       |
               +-------+     +-------------------------------+
                  | |        |        Block (Layer: l)       |
                  | |        |        ================       |
                  | |        |    +------+                   |
                  | |        |    | ^    |                   |
           <------+ +------------>| A[t] |--+                |
      Cl[t]          Rl[t]   |    |      |  |                |
                             |    +------+  |    +------+    |
                             |              +--->|      |    |
                             |                   | E[t] |---------->El[t]
                             |              +--->|      |    |
                             |    +------+  |    +------+    |
                             |    |      |  |                |
        Img/El-1[t]-------------->| A[t] |--+                |
                             |    |      |                   |
                             |    +------+                   |
                             |                               |
                             +-------------------------------+

--]]

local function block(l, L, iChannel, oChannel, vis,K,w,h)
   local inputs = {}

   --[[
       Inputs and outputs of ONE BLOCK
       Inputs: Image   -> A1 / El-1    -> Al
               Rl      -> Rl
               Total = 2

      Outputs: El
               Ah for layer 1
               Total = 1/2
   --]]

   -- Create input and output containers for nngraph gModule for ONE block
   table.insert(inputs, nn.Identity()())         -- A1 / El-1
   table.insert(inputs, nn.Identity()())         -- Rl

   local A
   local layer = tostring(l)
   if l == 1 then
      A = inputs[1]:annotate{name = 'X',
                    graphAttributes = gaA}
   else
      local nodeA = nn.Sequential()
      inputs[1]:annotate{name = 'E' .. layer-1}
      A = (inputs[1]
           - nodeA:add(SC(iChannel, oChannel, 3, 3, 1, 1, 1, 1))
                  :add(nn.ReLU())
                  :add(nn.SpatialMaxPooling(2, 2, 2, 2)))
                  :annotate{name = 'A' .. layer,
                            graphAttributes = gaA}
   end

   -- Get Rl
   local R = inputs[2]

   -- Predicted A
   local nodeAh = nn.Sequential()
   local Ah = (R:annotate{name = 'R' .. layer, graphAttributes = {
                          style = 'filled',
                          fillcolor = 'springgreen'}}
               - nodeAh:add(SC(oChannel, oChannel, 3, 3, 1, 1, 1, 1))
                       :add(nn.ReLU()))
                       :annotate{name = 'Ah' .. layer,
                                 graphAttributes = gaAh}

   -- Error between A and A hat
   local E = ({{A, Ah} - nn.CSubTable() - nn.ReLU(),
              {Ah, A} - nn.CSubTable() - nn.ReLU()}
             - nn.JoinTable(1, 3))
             :annotate{name = 'E' .. layer, graphAttributes = gaE}

   local g
   if l == 1 then
      -- For first layer return Ah for viewing
      local out = Ah - nn.View(oChannel*w*h) - nn.Linear(oChannel*w*h,K)
      - nn.Tanh()
      g = nn.gModule(inputs , {E, out})
   else
      g = nn.gModule(inputs, {E})
   end

   if vis then
      graph.dot(g.fg, 'Block_' .. l, 'graphs/Block_' .. l)
   end
   return g
end

local function stackBlocks(L, channels, vis, lstmLayer,K,w,h)
   --[[
       L -> Total number of layers
       Input and outputs in time series
       Inputs: Image   -> X      (1)
               RL+1    -> LSTM   (2)      This is always set to ZERO

               Cl[t-1] -> LSTM   (3l)     Cell state
               Hl[t-1] -> LSTM   (3l+1)   Hidden state
               El[t-1] -> LSTM   (3l+2)   Error
               Total = 3L+2

               {1, 3l-1, 3l, 3l+1}
      Outputs: Ah1, Cl, Hl, El
               Total = 3L+1
   --]]

   local inputs = {}
   local outputs = {}

   -- Create input and output containers for nngraph gModule for TIME SERIES
   for i = 1, (3*L+2) do
      table.insert(inputs, nn.Identity()():annotate{
         graphAttributes = {fontcolor = 'blue'}
      })
      local l = math.floor(i/3)
      if     i == 1 then   inputs[#inputs]:annotate{name = 'X[t]'}
      elseif i == 2 then   inputs[#inputs]:annotate{name = 'R'..(L+1)..'[t]'}
      elseif i%3 == 0 then inputs[#inputs]:annotate{name = 'C'..l..'[t-1]'}
      elseif i%3 == 1 then inputs[#inputs]:annotate{name = 'R'..l..'[t-1]'}
      elseif i%3 == 2 then inputs[#inputs]:annotate{name = 'E'..l..'[t-1]'} end
   end

--------------------------------------------------------------------------------
-- Get Rl
--------------------------------------------------------------------------------

   local upR, R
   local lstm = {}

   -- Calculate RL-1 -> RL-2 -> ... -> R1
   for l = L, 1, -1 do
      if l == L then
         upR = inputs[2] - nn.SpatialUpSamplingNearest(2)
      else
         upR = outputs[3*(l+1)] - nn.SpatialUpSamplingNearest(2)     -- Upsample prev Rl+1
      end
      R = {upR, inputs[3*l + 2]} - nn.JoinTable(1, 3)

      lstm = ({R, inputs[3*l], inputs[3*l+1]}
              - convLSTM:getModel(channels[l+1] + 2 * channels[l], channels[l], lstmLayer))
                        :annotate{name = 'LSTM ' .. l,
                                  graphAttributes = gaR}

      outputs[3*l-1] = (lstm - nn.SelectTable(1)):annotate{
         name = 'C'..l..'[t]', graphAttributes = {fontcolor = 'blue'}
      }                    -- Cell State
      outputs[3*l]   = (lstm - nn.SelectTable(2)):annotate{
         name = 'R'..l..'[t]', graphAttributes = {fontcolor = 'blue'}
      }                    -- Hidden state
   end

--------------------------------------------------------------------------------
-- Stack blocks to form the model for time t
--------------------------------------------------------------------------------
   for l = 1, L do
      local oChannel = channels[l]

      if l == 1 then          -- First layer block has E and Ah as output
         --                img,        Rl/Hl
         local E_Ah = ({inputs[1], outputs[3*l]}
                      - block(l, L, oChannel, oChannel, vis,K,w,h))
                      :annotate{name = '{E,Ah}'..l..'[t]', graphAttributes = gaE}

         local E, Ah = E_Ah:split(2)

         outputs[3*l+1] = E:annotate{name = 'E'..l..'[t]',
                                     graphAttributes = {
                                     style = 'filled',
                                     fillcolor = 'hotpink'}}

         outputs[1] = Ah:annotate{name = 'Prediction',
                                  graphAttributes = gaAh}
      else                    -- Rest of the blocks have only E as output
         local iChannel = 2 * channels[l-1]
                              -- El-1,           Rl/Hl
         outputs[3*l+1] = ({outputs[3*(l-1)+1], outputs[3*l]}
                          - block(l, L, iChannel, oChannel, vis,K,w,h))
                          :annotate{name = 'E'..l..'[t]', graphAttributes = gaE}
      end
   end

   return nn.gModule(inputs, outputs)
end

function prednet:getModel()
   local seq = self.seq
   local height = self.height
   local width  = self.width
   local L = self.layers
   local channels = self.channels
   local vis = self.saveGraph
   local K = self.K

   local prototype = stackBlocks(L, channels, vis, self.lstmLayer,K,width,height)

   if self.dev == 'cuda' then
      prototype:cuda()
   end

   local clones = {}
   for i = 1, seq do
      clones[i] = prototype:clone('weight', 'bias', 'gradWeight', 'gradBias')
   end

   local inputSequence = nn.Identity()()     -- Get input image as batch
   local RL_1 = nn.Identity()()              -- RL + 1 state which is always zero
   RL_1:annotate{name = "RL+1[t]"}

   local H0 = {}                             -- Default Rl - El pairs
   local H = {}                              -- States linking seq 1 -> 2 -> ...
   local outputs = {}                        -- Ah1

   for l = 1, 3*L do
      -- for annotation
      local styleColor = 'lightpink'
      local nodeName = sf('E%d[0]', l/3)
      if l % 3 == 1 then
         styleColor = 'springgreen'
         nodeName = sf('C%d[0]', (l+2)/3)
      elseif l % 3 == 2 then
         styleColor = 'burlywood'
         nodeName = sf('R%d[0]', (l+1)/3)
      end

      -- Link being created between input states to hidden states
      H0[l] = nn.Identity()()
      H[l] = H0[l]:annotate{name = nodeName,
                            graphAttributes = {
                            style = 'filled',
                            fillcolor = styleColor}}
   end

   -- eg for 5 grayscale images your input will be of dimension 5xhxw
   local splitInput = nn.SplitTable(1, 4)(inputSequence)

   for i = 1, seq do
      local inputFrame = nn.SelectTable(i)(splitInput):annotate{name = 'Input Frame #' .. i,
                                                                graphAttributes = {
                                                                style = 'filled',
                                                                fillcolor = 'gold1'}}

      -- Get Ah1 and all the El-Rl pairs as output from all the stacked layers
      local tempStates = ({inputFrame, RL_1, table.unpack(H)}
                           - clones[i]):annotate{name = 'Model Clone #' .. i,
                                                 graphAttributes = {
                                                 style = 'filled',
                                                 fillcolor = 'moccasin'}}

      -- Only Ah1 is sent as output
      outputs[i] = nn.SelectTable(1)(tempStates)       -- Send Ah to output
                                                :annotate{name = 'Prediction',
                                                          graphAttributes = gaAh}

      -- Rest of the {Cl, Hl, El} pairs are reused as {Cl[t-1], Hl[t-1], El[t-1]}
      if i < seq then
         for l = 1, 3*L do
            local styleColor = 'lightpink'
            local nodeName = sf('E%d[%d]', l/3, i)
            if l % 3 == 1 then
               styleColor = 'springgreen'
               nodeName = sf('C%d[%d]', (l+2)/3, i)
            elseif l % 3 == 2 then
               styleColor = 'burlywood'
               nodeName = sf('R%d[%d]', (l+1)/3, i)
            end

            -- Pass state values to next sequence
            H[l] = nn.SelectTable(l+1)(tempStates):annotate{
               name = nodeName, graphAttributes = {
                  style = 'filled', fillcolor = styleColor}}
         end
      end
   end

   local g = nn.gModule({inputSequence, RL_1, table.unpack(H0)}, outputs)

   if vis then
      graph.dot(prototype.fg, 'PredNet Model', 'graphs/predNet')
      graph.dot(g.fg, 'PredNet for whole sequence', 'graphs/wholeModel')
   end

   return g, clones[1]
end

return prednet
