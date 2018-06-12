import sys, os
sys.path.append(os.getcwd())
import json
from random import sample
from dataset import DataFromJSON as DJ 
from dataset import SeqDataFromJSON as SDJ 
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet50, alexnet
from torchvision import transforms
from generic_training import train, validate

from alexnet_lstm import ALSTM, ALSTM1, AFC

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

    train_data = data[:(len(data)*3)//4]

    test_data = data[(len(data)*3)//4:]

    # define transformation
    transformations = transforms.Compose([
            transforms.Scale(256),
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


def main():
    device = 'cuda'
    recurrent = False

    BATCH_SIZE = 128

    datapath = 'data/'
    train_loader, test_loader = gen_loaders(datapath, recurrent, BATCH_SIZE, 4)


    use_model, use_state = False, False
    net = AFC()

    lr = 0.0005
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    epoch = 0

    net = nn.DataParallel(net).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    train_args = OrderedDict()

    train_args['model'] = net
    train_args['rnn'] = False
    train_args['trainloader'] = train_loader
    train_args['testloader'] = test_loader
    train_args['batch_size'] = BATCH_SIZE
    train_args['criterion'] = criterion
    train_args['optimizer'] = optimizer
    train_args['target_accr'] = (100,)
    train_args['err_margin'] = (1.,)
    train_args['best_acc'] = (0,)
    train_args['topk'] = (1,)
    train_args['lr_decay'] = 0.8
    train_args['saved_epoch'] = 0 
    train_args['log'] = 'imitation_fc_bn.csv'
    train_args['pname'] = 'imitation_fc_bn_best.pth'
    train_args['cuda'] = (device == 'cuda')

    train(*train_args.values())


if __name__ == '__main__':
    main()
