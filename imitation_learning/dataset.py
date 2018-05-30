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

        # ret_act = torch.FloatTensor([0 if x != hot else 1 for x in range(8)])

        return (img_tensor, hot)


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
