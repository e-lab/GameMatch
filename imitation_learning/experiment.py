import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from utils.generic_training import train, validate
# from utils.generic_training_multi_lstm import train, validate

# from models.models_1room import AFC, ALSTM, ALSTM1
from models.basic_models import AFC, ALSTM
from models.models_2rooms_jun25 import ALSTM2rooms
from models.aac_lstm import BaseModelLSTM
from models.aac import BaseModel

from utils.imutils import gen_loaders

import random


configs = {'AFC': \
        {'model':AFC, \
        'GPU':'cuda:0', \
        'rec':False, \
        'name':'bn_fc'}, \
        'Conv':
        {'model':BaseModel, \
        'GPU':'cuda:0', \
        'rec':False, \
        'name':'basemodel'}, \
        'ALSTM': \
        {'model':ALSTM, \
        'GPU':'cuda:1', \
        'rec':True, \
        'hs': 1000, \
        'name':'bn_lstm_fc'}, \
        'ALSTM2rooms': \
        {'model':ALSTM2rooms, \
        'GPU':'cuda:2', \
        'rec':True, \
        'hs': (1024, 512), \
        'name':'bn_2lstm_fc'}, \
        'BaseLSTM':
        {'model':BaseModelLSTM, \
        'GPU':'cuda:3', \
        'rec':True, \
        'hs': 512, \
        'name':'baselstm'}
        }


def exp(name, device_id=0):
    # set the visible gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    config = configs[name]

    net = config['model']()
    
    device = 'cuda'

    BATCH_SIZE = 32 if name == 'AFC' or name == 'Conv' else 1

    datapath = 'data_2room/'
    train_loader, test_loader = gen_loaders(datapath, config['rec'], BATCH_SIZE, 4)


    lr = 0.0005
    epoch = 0

    net = nn.DataParallel(net).to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss().to(device)
    
    train_args = OrderedDict()

    train_args['model'] = net
    train_args['rnn'] = (name != 'AFC' and name != 'Conv')
    if name == 'AFC' or name == 'Conv':
        train_args['hidden_size'] = None
    else:
        train_args['hidden_size'] = config['hs']
    train_args['trainloader'] = train_loader
    train_args['testloader'] = test_loader
    train_args['batch_size'] = BATCH_SIZE
    train_args['criterion'] = criterion
    train_args['optimizer'] = optimizer
    train_args['target_accr'] = None    # (100,) 
    train_args['err_margin'] = (0.005,)
    train_args['best_acc'] = (0,)
    train_args['topk'] = (1,)
    train_args['lr_decay'] = 0.8
    train_args['saved_epoch'] = 0
    '''
    train_args['log'] = '1room_exp/1roomb_'+config['name']+'.csv'
    train_args['pname'] = '1room_exp/1roomb_'+config['name']+'.pth'
    '''
    train_args['log'] = '2room_exp/2room_'+config['name']+'_jun25.csv'
    train_args['pname'] = '2room_exp/2room_'+config['name']+'_jun25.pth'
    train_args['cuda'] = True

    train(*train_args.values())


def reinit(m):
    if type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif type(m) == nn.BatchNorm1d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# exp2 is used to find the optimal seqence-length
def exp2(name, seq_len, device_id='0'):
    print('******')
    print('sequence', seq_len)
    # set the visible gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    config = configs[name]

    net = config['model']()
    
    device = 'cuda'

    BATCH_SIZE = seq_len if name == 'AFC' or name == 'Conv' else 1

    datapath = 'data_2room/'
    train_loader, test_loader = gen_loaders(datapath, config['rec'], seq_len, BATCH_SIZE, 4)


    lr = 0.0005
    epoch = 0
    criterion = nn.CrossEntropyLoss().to(device)

    net = net.to(device)
    train_args = OrderedDict()


    for i in range(5):
        print('running experiment', i)
        torch.cuda.manual_seed(0)

        net.apply(reinit)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        
        train_args['model'] = net
        train_args['rnn'] = (name != 'AFC' and name != 'Conv')
        if name == 'AFC' or name == 'Conv':
            train_args['hidden_size'] = None
        else:
            train_args['hidden_size'] = config['hs']
        train_args['trainloader'] = train_loader
        train_args['testloader'] = test_loader
        train_args['batch_size'] = BATCH_SIZE
        train_args['criterion'] = criterion
        train_args['optimizer'] = optimizer
        train_args['target_accr'] = None    # (100,) 
        train_args['err_margin'] = (0.005,)
        train_args['best_acc'] = (0,)
        train_args['topk'] = (1,)
        train_args['lr_decay'] = 0.8
        train_args['saved_epoch'] = 0
        train_args['log'] = '../../stats/2rooms/2room_'+config['name']+'_jun28_seq' + str(seq_len) + '.csv'
        train_args['pname'] = '../../models/2rooms/2room_'+config['name']+'_jun28_seq' + str(seq_len) + '.pth'
        train_args['cuda'] = True

        print(train_args)

        train(*train_args.values())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-l', '--length', type=int, default=16)
    parser.add_argument('-d', '--device', type=str, default='0')

    args = parser.parse_args()

    seq_length = args.length

    device_id = args.device

    exp2("ALSTM", seq_length, device_id)
