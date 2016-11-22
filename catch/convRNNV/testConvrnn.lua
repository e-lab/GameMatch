gm = require 'toSeq'

ich , och , nHL, K, seq, nFW = 1, 1, 3, 1, 8, true
w, h , action = 10 , 10 , 3
batch = 64
model, single, linear = gm:getModel(ich, och, nHL, K, seq, nFW, w, h , action)

print(model)
x = torch.zeros(batch,seq,ich,10,10)
h = {}
for i = 1 , nHL do
   table.insert(h, torch.zeros(batch,och,10,10))
end
input = { x, table.unpack(h) }
print('input',input)
output = model:forward(input)

print('-- model output --')
print(output)

print('--- Single output ---')

x = torch.zeros(batch,ich,10,10)
h = {}
for i = 1 , nHL do
   table.insert(h, torch.zeros(batch,och,10,10))
end
tmpout = single:forward({x, table.unpack(h)})
print(tmpout)

print('--- Action single - linear output ---')
out    = linear:forward(x)
print({out})

