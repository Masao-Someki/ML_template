#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
#export CUDA_DEVICE_ORDER=PCI_BUS_ID

export PATH=$PATH:${PWD}/tools/commands
export PYTHONPATH=$PYTHONPATH:${PWD}/src/net:${PWD}/src/utils:${PWD}/src/writer

source tools/venv/bin/activate
