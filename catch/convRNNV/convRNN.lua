-- Original code from Karpaty char-rnn modified to convRNN
-- spk Nov 2016

require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
--nngraph.setDebug(true)

local backend = nn
local scNB = backend.SpatialConvolution:noBias()

function convRNN(inDim, outDim, kw, kh, st, pa, layerNum)

  local stw, sth = st, st
  local paw, pah = pa, pa
  local inputs = {}
  table.insert(inputs, nn.Identity()())

  for L = 1, layerNum do
    table.insert(inputs, nn.Identity()())
  end

  local outputs = {}
  for L = 1, layerNum do

    local x, prev_h
    prev_h = inputs[L+1]
    if L == 1 then
      x = inputs[1]
    else
      x = outputs[(L-1)]
    end

    local i2h , h2h
    if L == 1 then
      i2h = scNB(inDim, outDim, kw, kh, stw, sth, paw, pah)(x):annotate{name='i2h_'..L}
    else
      i2h = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(x):annotate{name='i2h_'..L}
    end
      h2h = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prev_h):annotate{name='h2h_'..L}
    local next_h = nn.Tanh()(nn.CAddTable(1,1){i2h, h2h})

    table.insert(outputs, next_h)
  end

  return nn.gModule(inputs, outputs)
end

return convRNN
