from vizdoom import *
import random
import time
# from imitation_learning.aac_base import BaseModel
from collections import OrderedDict
import itertools as it
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torch.autograd import Variable as V
from torchvision import transforms


# Pre-process state
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transformations = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
])


# Create and initialize Doom Game
game = DoomGame()
game.load_config("scenarios/my_way_home.cfg")
game.set_doom_scenario_path("scenarios/1roomfront.wad")
# game.set_doom_scenario_path("scenarios/1roomback.wad")
# game.set_doom_scenario_path("scenarios/2rooms.wad")
# game.set_doom_scenario_path("scenarios/3rooms.wad")

# june 21 enable freelook engine
game.add_game_args('+freelook 1')

game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_screen_format(ScreenFormat.RGB24)

game.set_window_visible(True)

game.init()

n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

# Action decoder
def decode_action(action):
    _, index = action.max(1)
    value = index
    decoded_action = [int(x) for x in bin(value)[2:].zfill(n)]
    decoded_action.reverse()

    return decoded_action


'''
changed on June 20

# Define and Load model weights
model_loadfile = "trained_models/2room.pth"
print("Loading model from: ", model_loadfile)
model = BaseModel(3, 2**5, 0, 0)
my_sd = torch.load(model_loadfile)
model.load_state_dict(my_sd['params'])
'''
from models.alexnet_lstm import AFC
model = AFC()
checkpoint = torch.load("1room_bn_fc.pth")
old_state = checkpoint['params']

new_state = OrderedDict()
for k, v in old_state.items():
    name = k[7:]
    new_state[name] = v

model.load_state_dict(new_state)
model = model.cuda()
model.eval()


# Start playing loop
episodes = 3
frame_repeat = 2
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        screen = state.screen_buffer
        screen = Image.fromarray(screen.astype('uint8'), 'RGB')
        screen = transformations(screen)
        screen = screen.unsqueeze(0).cuda()
        # print(screen.shape)
        action = model(screen)

        game.make_action(decode_action(action))
        # print("\treward:", reward)
        time.sleep(0.02)

    print("Result:", game.get_total_reward())
    time.sleep(2)

game.close()
