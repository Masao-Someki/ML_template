
import argparse

import torch
import numpy as np

from utils import get_config
from writer import Logger
from dataset import Dataset
from torch.utils.data import DataLoader
from loss import Loss

def train(args):
    # load config
    config = get_condfig(args.config)

    # set logger
    logger = Logger(args.log_name, 'train', 'val')

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set model and initialize other training settings
    net = Net(config.model)
    net.to(device)
    iter_count = 0
    optim = Optimizers(config.optim)
    optim.set_parameters(list(net.named_parameters()))
    criteria_before = 10000
    loss_fn = Loss()
    model_past = None

    # resume
    if args.resume is not None:
        dic = torch.load(args.resume, map_location=device)
        net.load_state_dict(dic['model'])
        iter_count = dic['iter_count']
        optim = dic['optim']
        criteria_before = dic['criteria']
        model_past = dic['model_path']

    # dataset and dataloaders
    datasets = {
            'train': Dataset(args.train_dir, device),
            'val'  : Dataset(args.val_dir, device)
    }
    dataloaders = {
            'train': DataLoader(datasets['train'],
                                batch_size=config.train.batch_size,
                                shuffle=config.train.shuffle),
            'val'  : DataLoader(datasets['val'],
                                batch_size=config.val.batch_size,
                                shuffle=config.val.shuffle)
    }

    # training!!
    for e in range(train.config.epoch):
        running_loss = []
        for batch in data_loaders['train']:
            # iter_count
            iter_count += 1

            # datas are already send to GPU.
            inputs, labels = batch

            # forward propagation
            out = net(inputs)

            # compute loss
            loss, _ = loss_fn(out, labels)

            # BP
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # log
            running_loss.append(loss.item())

            if iter_count % config.train.log_every:
                logger.train.figure(np.mean(running_loss), iter_count)

            # validation and save model
            if iter_count % config.train.save_every:
                val_loss = 0.0
                criterias = 0.0
                net.eval()
                with torch.no_grad():
                    for batch in dataloaders['val']:
                        inputs, labels = batch
                        out = net(inputs)
                        loss, criteria = loss_fn(out, labels)
                        val_loss.append(loss.item())
                        criterias.append(criteria)
                
                criteria = np.mean(criterias)
                logger.val.figure(np.mean(val_loss))
                logger.val.figure(criteria)

                if criteria < criteria_before:
                    logger.val.info('Criteria %f is smaller than criteria_before: %f'\
                            % (criteria, criteria_before))
                    model_name = os.path.join(args.model_dir, 'trained.%d.pt' % iter_count)

                    # remove past model to save memory
                    if model_past is not None:
                        os.remove(model_past)
                        logger.val.info('Removed existing best model at %s.' % model_past)

                    # save model
                    save_dic = {
                            'model': net.state_dict(),
                            'optim': optim,
                            'iter_count': iter_count,
                            'criteria': criteria,
                            'model_path': model_name
                    }
                    torch.save(save_dic, model_name)

    logger.train.info('Finished training model.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default=None, type=str,
                        help='Directory where training data is saved.')
    parser.add_argument('--val_dir', default=None, type=str,
                        help='Directory where validation data is saved.')
    parser.add_argument('--model_dir', default=None, type=str,
                        help='Directory where model files will be saved.')
    parser.add_argument('--log_name', default=None, type=str,
                        help='Log name.')
    parser.add_argument('--conf_path', default=None, type=str,
                        help='Path to the config file.')
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to the checkpoint from where to start training.')
    args = parser.parse_args()

    train(args)
