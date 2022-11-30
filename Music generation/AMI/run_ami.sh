#!/bin/bash

model="ami12"

# Genre
# music_genres="classical electronic rock jazz"
music_genres="classical emotion"

for music in $music_genres; do

echo $music

# Prepare features
python3 ami_prepare.py \
    --model $model \
    --sample_freq 12 \
    --key_shifts 7 \
    --midi_dir data/midi/ami/$music \
    --data_dir data/features/$model/$music \
    --model_dir models

done


# Train general model
music="classical"

python3 ami_train.py \
    --train_data_dir data/features/$model/$music  \
    --model_name_or_path models/${model}-template \
    --output_dir models/$model \
    --per_gpu_train_batch_size 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 


# Finetune model
music="emotion"

python3 ami_train.py \
    --train_data_dir data/features/$model/$music \
    --model_name_or_path models/$model \
    --output_dir models/$model-$music \
    --per_gpu_train_batch_size 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 3

# Generate
python3 ami_generate.py \
    --model_name_or_path models/$model-$music \
    --repetition_penalty 1.0 \
    --temperature 1.1 \
    --k 40 \
    --p .9 \
    --prompt '["<Q1>", "<BOS>"]' \
    --prompt_beats 8 \
    --generate_beats 128


python3 ami_generate.py \
    --model_name_or_path models/$model-$music \
    --repetition_penalty 1.0 \
    --temperature 1.1 \
    --k 40 \
    --p .9 \
    --prompt '["<Q2>", "<BOS>"]' \
    --prompt_beats 8 \
    --generate_beats 128

python3 ami_generate.py \
    --model_name_or_path models/$model-$music \
    --repetition_penalty 1.0 \
    --temperature 1.1 \
    --k 40 \
    --p .9 \
    --prompt '["<Q3>", "<BOS>"]' \
    --prompt_beats 8 \
    --generate_beats 128

python3 ami_generate.py \
    --model_name_or_path models/$model-$music \
    --repetition_penalty 1.0 \
    --temperature 1.1 \
    --k 40 \
    --p .9 \
    --prompt '["<Q4>", "<BOS>"]' \
    --prompt_beats 8 \
    --generate_beats 128

