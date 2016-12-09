-- Eugenio Culurciello
-- December 2016
-- test a trained deep Q learning neural network 

local base_path="/Users/eugenioculurciello/Desktop/ViZDoom/"
package.path = package.path .. ";"..base_path.."lua/vizdoom/?.lua"
require 'vizdoom.init'
require 'nn'
require 'image'

torch.setnumthreads(8)
torch.setdefaulttensortype('torch.FloatTensor')

local opt = {...}
opt.fpath = opt[1]
if not opt.fpath then print('missing arg #1: missing network file to test!') return end

print(opt.fpath)

-- load trained network:
local model = torch.load(opt.fpath)

local actions = {
    [1] = torch.Tensor({1,0,0}),
    [2] = torch.Tensor({0,1,0}),
    [3] = torch.Tensor({0,0,1})
}

-- Converts and down-samples the input image
local function preprocess(inImage)
  return image.scale(inImage, unpack(resolution))
end

-- Creates and initializes ViZDoom environment.
function initializeViZdoom(config_file_path)
    print("Initializing doom...")
    game = vizdoom.DoomGame()
    game:setViZDoomPath(base_path.."bin/vizdoom")
    game:setDoomGamePath(base_path.."scenarios/freedoom2.wad")
    game:loadConfig(config_file_path)
    game:setWindowVisible(opt.display)
    game:setMode(vizdoom.Mode.PLAYER)
    game:setScreenFormat(vizdoom.ScreenFormat.GRAY8)
    game:setScreenResolution(vizdoom.ScreenResolution.RES_640X480)
    game:init()
    print("Doom initialized.")
    return game
end

while true do
  -- Create Doom instance
  local game = initializeViZdoom(config_file_path)

  -- Reinitialize the game with window visible
  game:setWindowVisible(true)
  game:setMode(vizdoom.Mode.ASYNC_PLAYER)
  game:init()

  for i = 1, opt.episodesWatch do
      game:newEpisode()
      while not game:isEpisodeFinished() do
          local state = preprocess(game:getState().screenBuffer:float():div(255))
          local best_action_index = getBestAction(state)

          -- Instead of make_action(a, frame_repeat) in order to make the animation smooth
          game:makeAction(actions[best_action_index])
          for j = 1, opt.frameRepeat do
              game:advanceAction()
          end
      end

      -- Sleep between episodes:
      sys.sleep(1)
      local score = game:getTotalReward()
      print("Total score: ", score)
  end
end
game:close()
