#!/bin/bash

model="ami12"

# Train general model
music="emotion"

python3 ami_train.py \
    --train_data_dir data/features/$model/$music  \
    --model_name_or_path models/${model}-pretrained3 \
    --output_dir models/${model}-pre3-emopia1-lr \
    --per_gpu_train_batch_size 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 1

