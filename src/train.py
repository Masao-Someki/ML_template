# -*- coding: <encoding name> -*-

# training script
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import Net
from dataset import Dataset
from loss import Loss
from optim import Optimizers
from writer import Logger
from utils import get_config

def train(args):
    # load config
    config = get_config(args.conf_path)

    # logger
    logger = Logger(args.log_name, 'train', 'val')

    # training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # trainind settings and model
    net = Net(config.model)
    net.to(device)
    iter_count = 0
    optim = Optimizers(config.optim)
    optim.set_parameters(list(net.named_parameters()))
    criteria_before = 0
    past_model = ''
    
    # resume
    if args.resume is not None:
        dic = torch.load(args.resume)
        net.load_state_dict(dic['model'])
        iter_count = dic['iter_count']
        optim = dic['optim']
        criteria_before = dic['criteria']
        past_model = dic['path']

    # dataset
    datasets = {'train': Dataset(args.train_dir, device),
                'val'  : Dataset(args.val_dir, device)
                }
    data_loaders = {'train': DataLoader(datasets['train'],
                                        batch_size=config.train.batch_size,
                                        shuffle=True),
                    'val'  : DataLoader(datasets['val'],
                                        batch_size=config.val.batch_size,
                                        shuffle=True)
                    }

    # loss function
    loss_fn = Loss()

    sys.exit(0)

    # training!
    logger.train.info('Start training from iteration %d' % iter_count)
    for e in range(config.train.epoch):
        # iter for batch
        net.train()
        losses = []
        for batch in data_loaders['train']:
            # iter_count
            iter_count += 1

            # forward propagation
            out = net(batch)

            # compute loss
            loss, _ = loss_fn(out)

            # back propagation
            optim.zero_grad()
            loss.backward()
            optim.step()

            # log
            logger.train.figure(loss, iter_count)
            losses.append(loss.cpu().detach().numpy())

        # log
        logger.train.info('Loss for epoch %d : %.5f' % (e, np.mean(losses)))

        # Validation
        logger.val.info('Start validation at epoch %d' % e)
        net.eval()
        losses = []
        with torch.no_grad():
            for batch in data_loaders['val']:
                # forward propagation
                out = net(batch)

                # compute loss
                loss, criteria = loss_fn(out)

                # log
                losses.append(loss.cpu().detach().numpy())
        
        # log
        logger.val.info('Validation loss at epoch %d: %.5f' % (e, np.mean(losses)))

        # save model with best criteria
        if criteria < criteria_before:
            logger.val.info('Passed criteria (%f < %f), saving best model...' \
                    % (criteria, criteria_before))
            
            # remove the existing past model.
            if not past_model == '':
                os.remove(past_model)
                logger.val.info('Found existing model at %s. Removed this file.' % past_model)

            # build dict
            save_file = os.path.join(args.model_dir, 'trained.%d.pt' % iter_count)
            save_dic = {
                    'model': net.state_dict(),
                    'iter_count': iter_count,
                    'optim': optim,
                    'criteria': criteria,
                    'path': save_file
                }
            torch.save(save_dic, save_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default=None, type=str,
                        help='Path to the training data.')
    parser.add_argument('--val_dir', default=None, type=str,
                        help='Path to the validation data')
    parser.add_argument('--conf_path', default=None, type=str,
                        help='Path to the config file')
    parser.add_argument('--model_dir', default=None, type=str,
                        help='Path to the directory where trained model will be saved.')
    parser.add_argument('--log_name', default=None, type=str,
                        help='Name log file will be saved')
    args = parser.parse_args()

    train(args)
