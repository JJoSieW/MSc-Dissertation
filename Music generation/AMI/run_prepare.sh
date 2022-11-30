#!/bin/bash

model="ami12"

# Genre
#music_genres="classical emotion"
music_genres="mix"

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


