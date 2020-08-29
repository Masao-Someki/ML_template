#!/bin/bash

stage=012345

exp_dir=exp
data_dir=data
model_dir=model

log_name=demo
checkpoint=


. parse_options.sh


#####################################
############# stage r ###############

if echo ${stage} | grep -q -r r; then
	rm -r log/log/${log_name}
	rm -r log/tbx/${log_name}
fi

if echo ${stage} | grep -q -r m; then
	rm -r ${model_dir}/${log_name}.*.pt
fi

if echo ${stage} | grep -q -r d; then
	rm -r ${data_dir}/train/*
	rm -r ${data_dir}/val/*
fi



#####################################
############# stage 0 ###############

# stage 0
# data download

if echo ${stage} | grep -q -r 0; then
	if [ ! -e ./data/MNIST ]; then
		python src/prepro.py \
			--train_dir ./data
	fi
fi

#####################################
############ stage 1 ################

# data preprocessing

######################################
############ stage 2 ################@

# training
if exho ${stage} | grep -q -r 2; then
python src/train.py \
	--train_dir ./data/MNIST/processed/train/training.pt \
	--val_dir ./data/MNIST/processed/val/test.pt \
	--model_dir ./model/demo \
	--log_name demo \
	--conf_path ./config/demo.conf 
fi

######################################
############ stage 3 ################3

# evaluate best model with test samples


######################################
############ stage 4 #################

# export on onnx model to speed up inference
