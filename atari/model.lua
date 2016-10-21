-- Eugenio Culurciello
-- October 2016
-- Deep Q learning code

require 'nn'

function createModel (n_actions, stacked_frames)

   local net = nn.Sequential()
   net:add(nn.SpatialConvolution(stacked_frames,32,7,7,3,3))
   net:add(nn.ReLU())

   net:add(nn.SpatialConvolution(32,64,5,5,2,2))
   net:add(nn.ReLU())

   net:add(nn.SpatialConvolution(64,64,3,3,1,1))
   net:add(nn.SpatialMaxPooling(2,2,2,2))
   net:add(nn.ReLU())

   -- net:add(nn.SpatialAveragePooling(4,4))

   net:add(nn.View(64*4*4))
   net:add(nn.Linear(64*4*4, 32))
   net:add(nn.ReLU())
   net:add(nn.Linear(32, n_actions))

   local criterion = nn.MSECriterion()

   return net, criterion
end