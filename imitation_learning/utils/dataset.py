# utils.dataset
# Customize dataloaders used in training and testing
# Author: Ruihang Du
# email: du113@purdue.edu

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import json

from PIL import Image

# prepare dataset from the json file
class DataFromJSON(Dataset):
    def __init__(self, data_path=None, data_list=None, transforms=None):
        self.data_path = data_path
        self.data_list = data_list

        if data_path and not data_list:
            # if we don't already have a list of tensor data, we load them
            # the name of stored frames and other important information 
            # will be read in from the json file
            with open(data_path + 'data.json', 'r') as datafile:
                # loads will create a list of dictionaries
                self.data = json.loads(json.load(datafile))
        if data_list:
            # otherwise, just use the preexisting data list
            self.data = data_list

        self.transforms = transforms

    def __getitem__(self, index):
        item = self.data[index]
        tfname = self.data_path + 'screens/'+item['screen']
        # the img_tensor will be a numpy array
        img_arr = torch.load(tfname)
        img = Image.fromarray(img_arr.astype('uint8'), 'RGB')
        if self.transforms:
            img_tensor = self.transforms(img)

        # action should be a list of numbers
        action = item['action']

        # this encodes the action from a 'binary string' to an integer
        hot = int(sum([action[i]*(2**i) for i in range(len(action))]))
        
        '''
        wrong, could have combinations
        # june 21: use common encoding of the result
        # there are six available actions to take
        hot = 0
        for i, a in enumerate(action, 1):
            if abs(a - 0) > 1e-4:
                hot = i
        '''

        # ret_act = torch.FloatTensor([0 if x != hot else 1 for x in range(8)])

        return (img_tensor, hot)


    def __len__(self):
        return len(self.data)


# prepare a dataset containing sequence of data points
# used for networks with LSTM
class SeqDataFromJSON(Dataset):
    def __init__(self, data_path=None, data_list=None, seq_len=32, transforms=None):
        self.data_path = data_path
        self.data_list = data_list

        if data_path and not data_list:
            with open(data_path + 'data.json', 'r') as datafile:
                # loads will create a list of dictionaries
                self.data = json.loads(json.load(datafile))
        if data_list:
            self.data = data_list

        # now we create list of sequences
        self.data = [self.data[i:i+seq_len] for i in range(0, len(self.data)-seq_len+1, seq_len)]
        
        self.seq_len = seq_len
        
        self.transforms = transforms

    def __getitem__(self, index):
        islice = self.data[index]
        imgs = []
        actions = []
        for item in islice:
            tfname = self.data_path + 'screens/'+item['screen']
            # the img_tensor will be a numpy array
            img_arr = torch.load(tfname)
            img = Image.fromarray(img_arr.astype('uint8'), 'RGB')
            if self.transforms:
                img_tensor = self.transforms(img)

            imgs.append(img_tensor)

            # action should be a list of numbers
            action = item['action']
            hot = int(sum([action[i]*(2**i) for i in range(len(action))]))
            '''
            # june 21: use common encoding of the result
            # there are six available actions to take
            hot = 0
            for i, a in enumerate(action, 1):
                if abs(a - 0) > 1e-4:
                    hot = i
            '''

            actions.append(hot)

        imgs = torch.stack(imgs)
        actions = torch.LongTensor(actions)

        return (imgs, actions)


    def __len__(self):
        return len(self.data)
