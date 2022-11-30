#!/bin/bash

#model="ami12-finetuned3"
#model="ami12-pretrained5"
#model="ami12-myemotion-epoch2-lr"
#model="ami12-emotion-epoch3"
#model="ami12-mix-epoch2"
model="ami12-pre3-myemo1"

# Emotion
emotion_classes="Q1 Q2 Q3 Q4"
#emotion_classes="Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4"
#emotion_classes="Q1 Q1 Q1 Q1 Q1"

echo $model
for emotion in $emotion_classes; do
echo $emotion
# Generate
python3 ami_generate.py \
    --model_name_or_path models/$model \
    --repetition_penalty 1.1 \
    --temperature 1.1 \
    --k 30 \
    --p .90 \
    --prompt '["<'$emotion'>", "<BOS>"]' \
    --prompt_beats 8 \
    --generate_beats 64

done





