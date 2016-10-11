require 'nn'

function createModel (n_actions)

   local net = nn.Sequential()
   net:add(nn.SpatialConvolution(3,32,9,9,4,4))
   net:add(nn.ReLU())

   net:add(nn.SpatialConvolution(32,64,5,5,2,2))
   net:add(nn.SpatialMaxPooling(2,2,2,2))
   net:add(nn.ReLU())

   net:add(nn.SpatialConvolution(64,64,3,3,1,1))
   net:add(nn.SpatialMaxPooling(2,2,2,2))
   net:add(nn.ReLU())

   net:add(nn.View(64*5*3))
   net:add(nn.Linear(64*5*3, 512))
   net:add(nn.ReLU())
   net:add(nn.Linear(512, n_actions))

   local criterion = nn.MSECriterion()

   return net, criterion
end