# -*- coding: utf-8 -*-

# script for preprocess

import argparse

import torch
import torchvision

from writer import Logger

def download(path):
    if path is None:
        path = './'
    data = torchvision.datasets.MNIST(root=path, download=True)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default=None, type=str,
                        help='Path to the directory where the training data will be stored')
    args = parser.parse_args()

    # logger
    logger = Logger('', 'preprocess')

    # download dataset
    download(args.train_dir)
    logger.preprocess.info('Finished downloading datasets!')

