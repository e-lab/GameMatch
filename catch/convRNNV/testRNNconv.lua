torch.setdefaulttensortype('torch.FloatTensor')
require 'nn'
opt = {}
opt.fw = true
gm = require 'RNNconv'

n , d , nHL, K, T, nFW = 2, 4, 2, 1, 8, 3
w, h , action = 10 , 10 , 3
batch = 64
model, single = gm.getModel(n, d, nHL, K, T, nFW, batch, w, h)

print(model)
x = torch.zeros(batch,T,n,10,10)
h = {}
for i = 1 , nHL do
   table.insert(h, torch.zeros(batch,d,10,10))
end
input = { x, table.unpack(h) }
print('input',input)
output = model:forward(input)

print('-- model output --')
print(output)

print('--- Single output ---')

x = torch.zeros(batch,n,10,10)
h = {}
for i = 1 , nHL do
   table.insert(h, torch.zeros(batch,d,10,10))
end
tmpout = single:forward({x, table.unpack(h)})
print(tmpout)

