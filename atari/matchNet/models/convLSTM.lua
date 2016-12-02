-- First written by Sangpil Kim
-- Notation is from https://github.com/oxford-cs-ml-2015/practical6/blob/master/LSTM.lua
-- ConvLSTM with nngraph
-- August 2016
local convLSTM = {}

function convLSTM:getModel(inDim, outDim, lstmLayer)
   local sc = nn.SpatialConvolution
   local scNB = nn.SpatialConvolution:noBias()
   local sg = nn.Sigmoid
   local kw, kh  = 3, 3
   local stw, sth = 1, 1
   local paw, pah = 1, 1
   local n = lstmLayer
   -- Input  is 1+ 2*#Layer
   -- Output is 1+ 2*#Layer
   local inputs = {}
   table.insert(inputs, nn.Identity()():annotate{
      name = 'x[t]', graphAttributes = {fontcolor = 'blue'}}) -- X
   for l = 1, n do
      table.insert(inputs, nn.Identity()():annotate{
         name = 'c[t-1]', graphAttributes = {fontcolor = 'blue'}}) -- Cell
      table.insert(inputs, nn.Identity()():annotate{
         name = 'h[t-1]', graphAttributes = {fontcolor = 'blue'}}) -- Hidden state
   end

   local x
   local outputs = {}
   for l = 1,n do
      -- Container for previous C and H
      local prevH = inputs[l*2+1]
      local prevC = inputs[l*2]
      -- Get input
      if l == 1 then
         x = inputs[1]
      else
         -- Get x from bottom layer as input
         x = outputs[(l-1)*2]
      end
      --Convolutions
      local i2Ig, i2Fg, i2Og, i2It
      if l == 1 then
         i2Ig = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x):annotate{
            name = 'W_i x[t]', graphAttributes = {fontcolor = 'red'}}
         i2Fg = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x):annotate{
            name = 'W_f x[t]', graphAttributes = {fontcolor = 'red'}}
         i2Og = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x):annotate{
            name = 'W_o x[t]', graphAttributes = {fontcolor = 'red'}}
         i2It = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x):annotate{
            name = 'W_c x[t]', graphAttributes = {fontcolor = 'red'}}
      else
         i2Ig = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
         i2Fg = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
         i2Og = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
         i2It = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
      end

      local h2Ig, h2Fg, h2Og, h2It
      h2Ig = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevH):annotate{
         name = 'W_i h[t-1]', graphAttributes = {fontcolor = 'red'}}
      h2Fg = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevH):annotate{
         name = 'W_f h[t-1]', graphAttributes = {fontcolor = 'red'}}
      h2Og = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevH):annotate{
         name = 'W_o h[t-1]', graphAttributes = {fontcolor = 'red'}}
      h2It = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevH):annotate{
         name = 'W_c h[t-1]', graphAttributes = {fontcolor = 'red'}}

      local ig = nn.CAddTable()({i2Ig, h2Ig})
      local fg = nn.CAddTable()({i2Fg, h2Fg})
      local og = nn.CAddTable()({i2Og, h2Og})
      local it = nn.CAddTable()({i2It, h2It})

      -- Gates
      local inGate = sg()(ig):annotate{
         name = 'i[t]', graphAttributes = {fontcolor = 'green'}}
      local fgGate = sg()(fg):annotate{
         name = 'f[t]', graphAttributes = {fontcolor = 'green'}}
      local ouGate = sg()(og):annotate{
         name = 'o[t]', graphAttributes = {fontcolor = 'green'}}
      local inTanh = nn.Tanh()(it):annotate{
         name = 'c\'[t]', graphAttributes = {fontcolor = 'green'}}
      -- Calculate Cell state
      local nextC = nn.CAddTable()({
         nn.CMulTable()({fgGate, prevC}),
         nn.CMulTable()({inGate, inTanh})
      }):annotate{name = 'c[t]', graphAttributes = {fontcolor = 'blue'}}
      -- Calculate output
      local out = nn.CMulTable()({ouGate, nn.Tanh()(nextC)}):annotate{
         name = 'h[t]', graphAttributes = {fontcolor = 'blue'}}

      table.insert(outputs, nextC)
      table.insert(outputs, out)
   end

   -- Extract output
   local lastH = outputs[#outputs]

   local g = nn.gModule(inputs, outputs)
   graph.dot(g.fg, 'LSTM', 'graphs/LSTM')
   return g
end

return convLSTM
