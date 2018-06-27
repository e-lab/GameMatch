import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import json

from PIL import Image

class DataFromJSON(Dataset):
    def __init__(self, data_path=None, data_list=None, transforms=None):
        self.data_path = data_path
        self.data_list = data_list

        if data_path and not data_list:
            with open(data_path + 'data.json', 'r') as datafile:
                # loads will create a list of dictionaries
                self.data = json.loads(json.load(datafile))
        if data_list:
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
        '''
        # the reward should be a single number
        reward = item['reward']
        
        # return state
        return (img_tensor, action, reward)
        '''
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

'''
if __name__ == '__main__':
    # define transformation
    transformations = transforms.Compose([transforms.ToTensor()])

    # create dataset instance
    custom_data = DataFromJSON(data_path='data/', transforms=transformations)

    # use the dataloader as you would with other datasets
    dataloader = torch.utils.data.DataLoader(dataset=custom_data, \
                                                    batch_size=1,shuffle=False)

    for i, data in enumerate(dataloader):
        if i > 0: break
        img, action = data
        # since the original action is a list of torch double tensors, turn it to a tensor
        # action = torch.stack(action).float().squeeze()
        # cast reward to integer tensor
        # reward = reward.int()

        print('img', type(img))
        print('action', action)
        print('reward', reward)
'''
