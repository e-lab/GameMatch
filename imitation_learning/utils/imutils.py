# code that creates dataloaders for training and testing
# Author: Ruihang Du
# email: du113@purdue.edu

import json
from random import sample
from utils.dataset import DataFromJSON as DJ 
from utils.dataset import SeqDataFromJSON as SDJ 
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms


# generate the loaders for training and test data for simple networks
def gen_loaders(path, recurrent, seq_len, BATCH_SIZE, NUM_WORKERS):
    # whether the model has an recurrent layer
    Gen = SDJ if recurrent else DJ
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    with open(path + 'data.json', 'r') as datafile:
        # loads will create a list of dictionaries
        data = json.loads(json.load(datafile))

    print('total', len(data), 'data')

    train_data = data[:(len(data)*3)//4]

    print('number of training data:', len(train_data))

    test_data = data[(len(data)*3)//4:]

    print('number of testing data:', len(test_data))

    # define transformation
    transformations = transforms.Compose([
            transforms.Scale(256),
            # transforms.Resize(256),  
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
    ])

    if recurrent:
        train_set = Gen(data_path=path, data_list=train_data, seq_len=seq_len, transforms=transformations)
    else:
        train_set = Gen(data_path=path, data_list=train_data, transforms=transformations)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, \
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=NUM_WORKERS, 
                                                    pin_memory=True)

    if recurrent:
        test_set =  Gen(data_path=path, data_list=test_data, seq_len=seq_len, transforms=transformations)
    else:
        test_set =  Gen(data_path=path, data_list=test_data, transforms=transformations)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, \
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=NUM_WORKERS, 
                                                    pin_memory=True)

    
    return (train_loader, test_loader)

