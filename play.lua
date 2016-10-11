-- Eugenio Culurciello
-- October 2016

-- gd = require "gd"
require 'qtwidget' -- for keyboard interaction

if not dqn then
    require "initenv"
end


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Play video game:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', 'alewrap', 'name of training framework')
cmd:option('-env', 'breakout', 'name of environment to use')
cmd:option('-game_path', 'roms/', 'path to environment file (ROM)')
cmd:option('-env_params', 'useRGB=true', 'string of environment parameters')
cmd:option('-pool_frms', 'max', 'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 8, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-gif_file', '', 'GIF path to write session screens')
cmd:option('-csv_file', '', 'CSV path to write session data')

cmd:option('-zoom', '4', 'zoom window')

cmd:text()

local opt = cmd:parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')
-- torch.manualSeed(opt.seed)

local qtimer = qt.QTimer()

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

-- file names from command line
-- local gif_filename = opt.gif_file

-- start a new game
local screen, reward, terminal = game_env:newGame()

-- compress screen to JPEG with 100% quality
-- local jpg = image.compressJPG(screen:squeeze(), 100)
-- create gd image from JPEG string
-- local im = gd.createFromJpegStr(jpg:storage():string())
-- convert truecolor to palette
-- im:trueColorToPalette(false, 256)

-- write GIF header, use global palette and infinite looping
-- im:gifAnimBegin(gif_filename, true, 0)
-- write first frame
-- im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)

-- remember the image and show it first
-- local previm = im
-- local win = image.display({image=screen})

-- Create a window for displaying output frames
win = qtwidget.newwindow(opt.zoom*screen:size(4), opt.zoom*screen:size(3),'EC game engine')

local keyPress = 2

print("Started playing...")

-- play one episode (game)
function main()
-- while not terminal do
    -- if action was chosen randomly, Q-value is 0
    -- agent.bestq = 0
    
    -- choose the best action
    local action_index = keyPress --2+torch.random(2)
    -- local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

    -- play game in test mode (episodes don't end when losing a life)
    -- reward = score added, terminal = end of game (all lives gone)
    screen, reward, terminal = game_env:step(game_actions[action_index], false)
    -- print(reward, terminal)

    -- display screen
    image.display({image=screen, win=win, zoom=opt.zoom})

    -- create gd image from tensor
    -- jpg = image.compressJPG(screen:squeeze(), 100)
    -- im = gd.createFromJpegStr(jpg:storage():string())
    
    -- use palette from previous (first) image
    -- im:trueColorToPalette(false, 256)
    -- im:paletteCopy(previm)

    -- write new GIF frame, no local palette, starting from left-top, 7ms delay
    -- im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)
    -- remember previous screen for optimal compression
    -- previm = im

end

-- end GIF animation and close CSV file
-- gd.gifAnimEnd(gif_filename)
-- print("Finished playing, close window to exit!")


-- game controls
print('Game controls: left / right')
qtimer.interval = 10
qtimer.singleShot = false
qt.connect(qtimer,'timeout()', main)

local prevState = true
qt.connect(win.listener,
         'sigKeyPress(QString, QByteArray, QByteArray)',
         function(_, keyValue)
            if keyValue == 'Key_Right' then
                keyPress = 3
            elseif keyValue == 'Key_Left' then
                keyPress = 4
            elseif keyValue == 'Key_X' then
                keyPress = 1
            else 
                keyPress = 2
            end
            qtimer:start()
         end)
qt.connect(win.listener,
         'sigKeyRelease(QString, QByteArray, QByteArray)',
         function(_, keyValue) keyPress = 2 end)
qtimer:start()
