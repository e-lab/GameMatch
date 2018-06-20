import sys, os
sys.path.append(os.getcwd())

import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from generic_training import train, validate

from alexnet_lstm import AFC, ALSTM, ALSTM1

from imutils import gen_loaders


configs = {'AFC': \
        {'model':AFC, \
        'GPU':'cuda:0', \
        'rec':False, \
        'name':'bn_fc'}, \
        'ALSTM': \
        {'model':ALSTM, \
        'GPU':'cuda:1', \
        'rec':True, \
        'hs': 1000, \
        'name':'bn_lstm_fc'}, \
        'ALSTM1': \
        {'model':ALSTM1, \
        'GPU':'cuda:2', \
        'rec':True, \
        'hs': 8, \
        'name':'bn_lstm'}
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)

    args = parser.parse_args()

    name = args.model

    config = configs[name]

    net = config['model']()
    
    device = 'cuda'

    BATCH_SIZE = 128 if name == 'AFC' else 1

    datapath = 'data_HGS/'
    train_loader, test_loader = gen_loaders(datapath, config['rec'], BATCH_SIZE, 4)


    lr = 0.0005
    epoch = 0

    net = nn.DataParallel(net).to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss().to(device)
    
    train_args = OrderedDict()

    train_args['model'] = net
    train_args['rnn'] = (name != 'AFC')
    if name == 'AFC':
        train_args['hidden_size'] = None
    else:
        train_args['hidden_size'] = config['hs']
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
    train_args['log'] = 'hgs_'+config['name']+'.csv'
    train_args['pname'] = 'hgs_'+config['name']+'.pth'
    train_args['cuda'] = True

    train(*train_args.values())


if __name__ == '__main__':
    main()

