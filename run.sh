#!/bin/bash

stage=012345

#####################################
############# stage 0 ###############

# stage 0
# data download
if [ ! -e ./data/MNIST ]; then
	python src/prepro.py \
		--train_dir ./data
fi


#####################################
############ stage 1 ################

# data preprocessing


######################################
############ stage 2 ################@

# training
python src/train.py \
	--train_dir ./data/MNIST/processed/training.pt \
	--val_dir ./data/MNIST/processed/test.pt \
	--conf_path ./config/demo.conf \
	--model_dir ./model \
	--log_name mnist_test


######################################
############ stage 3 ################3

# evaluate best model with test samples


######################################
############ stage 4 #################

# export on onnx model to speed up inference

