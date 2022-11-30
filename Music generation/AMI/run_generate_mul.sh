#!/bin/bash

#model="ami12-finetuned3"
#model="ami12-pretrained5"
#model="ami12-myemotion-epoch2-lr"
#model_classes="ami12-emotion-epoch3"
model_classes=" ami12-pretrained3"

#model_classes="ami12-pre3-emopia1-lr"
#model_classes="ami12-pre3-emopia1 ami12-pre3-emopia1-lr ami12-pre3-myemo1 ami12-pre3-myemo1-lr"

# Emotion
#emotion_classes="Q1 Q2 Q3 Q4 Q1 Q2 Q3 Q4"
emotion_classes="Q1 Q2 Q3 Q4"
#emotion_classes="Q1 Q1 Q1 Q1 Q1"

for ((i=1; i<=15; i++))
do
  for model in $model_classes; do
      echo $model
      for emotion in $emotion_classes; do
        echo $emotion
        # Generate
        python3 ami_generate.py \
            --model_name_or_path models/$model \
            --repetition_penalty 1.0 \
            --temperature 1.0 \
            --k 35 \
            --p .95 \
            --prompt '["<'$emotion'>", "<BOS>"]' \
            --prompt_beats 8 \
            --generate_beats 90
      
      done
  done
done





