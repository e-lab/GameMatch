gm = require 'toSeq'

ich , och , nHL, K, seq, nFW = 3, 2, 3, 1, 8, true
w, h , action = 10 , 10 , 3
batch = 32
model = gm:getModel(ich, och, nHL, K, seq, nFW, w, h , action)
print(model)
x = torch.zeros(seq,batch,3,10,10)
h = {}
for i = 1 , nHL do
   table.insert(h, torch.zeros(batch,2,10,10))
end
input = { x, table.unpack(h) }
output = model:forward(input)

print(output)
