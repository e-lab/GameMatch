import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
# from utils.generic_training import train, validate
from utils.generic_training_multi_lstm import train, validate

# from models.models_1room import AFC, ALSTM, ALSTM1
from models.basic_models import AFC, ALSTM
from models.models_2rooms_jun25 import ALSTM2rooms
from models.aac_lstm import BaseModelLSTM
from models.aac import BaseModel

from utils.imutils import gen_loaders


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-d', '--device', type=str, default='0')

    args = parser.parse_args()

    name = args.model

    device_id = args.device

    exp(name, device_id)
