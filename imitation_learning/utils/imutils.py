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
from utils.generic_training import train, validate

# from models.alexnet_lstm import ALSTM, ALSTM1, AFC

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
    # device = 'cuda'
    devices = [torch.device('cuda:'+str(x)) for x in range(4)]

    recurrent = True

    BATCH_SIZE = 128

    datapath = 'data_HGS/'
    train_loader, test_loader = gen_loaders(datapath, recurrent, BATCH_SIZE, 4)


    use_model, use_state = False, False

    available_models = [AFC, ALSTM, ALSTM1]
    datafiles = ['hgs_bn_fc.csv', 'hgs_bn_lstm_fc.csv', 'hgs_bn_lstm.csv']
    modelfiles = ['hgs_bn_fc.pth', 'hgs_bn_lstm_fc.pth', 'hgs_bn_lstm.pth']

    lr = 0.0005
    epoch = 0

    for i, Net in enumerate(available_models):
        device = devices[i]

        if Net == AFC:
            train_loader, test_loader = gen_loaders(datapath, False, BATCH_SIZE, 4)
        
        net = nn.DataParallel(Net()).to(device)

        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        criterion = nn.CrossEntropyLoss().to(device)

        train_args = OrderedDict()

        train_args['model'] = net 
        train_args['rnn'] = (Net != AFC)
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
        train_args['log'] = datafiles[i]
        train_args['pname'] = modelfiles[i] 
        train_args['cuda'] = True

        train(*train_args.values())


if __name__ == '__main__':
    main()
