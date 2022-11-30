#!/bin/bash

model="ami12"


# Train general model
music="classical"

python3 ami_train.py \
    --train_data_dir data/features/$model/$music  \
    --model_name_or_path models/${model}-pretrained5 \
    --output_dir models/${model}-pretrained6 \
    --per_gpu_train_batch_size 2 \
    --learning_rate 1e-4 \
    --num_train_epochs 1  



# Note:
# old pretrained5 -> seven epochs  8

