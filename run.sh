#!/bin/bash

stage=012345

#####################################
############# stage 0 ###############

# stage 0
# data download
python src/prepro.py \
	--train_dir ./data


#####################################
############ stage 1 ################

# data preprocessing


######################################
############ stage 2 ################@

# training
#python train.py


######################################
############ stage 3 ################3

# evaluate best model with test samples


######################################
############ stage 4 #################

# export on onnx model to speed up inference

