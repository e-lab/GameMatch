# import sys, os
# sys.path.append(os.getcwd())
import json
from random import sample
from utils.dataset import DataFromJSON as DJ 
from utils.dataset import SeqDataFromJSON as SDJ 
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms


def gen_loaders(path, recurrent, BATCH_SIZE, NUM_WORKERS):
    Gen = SDJ if recurrent else DJ
    # Data loading code
    # traindir = os.path.join(path, 'train')
    # valdir = os.path.join(path, 'val')
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
            transforms.Resize(256),  # transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
    ])

    train_set = Gen(data_path=path, data_list=train_data, transforms=transformations)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, \
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=NUM_WORKERS, 
                                                    pin_memory=True)

    test_set =  Gen(data_path=path, data_list=test_data, transforms=transformations)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, \
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=NUM_WORKERS, 
                                                    pin_memory=True)

    
    return (train_loader, test_loader)

