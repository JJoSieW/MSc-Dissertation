#!/bin/bash
#$ -l h_rt=01:00:00  #time needed
#$ -l gpu=2
#$ -l rmem=50G
#$ -P rse
#$ -q rse.q
#$ -o Output/sample.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M myname@sheffield.ac.uk
#$ -m ea # email me when if finished or aborted
#$ -cwd # Run job from current directory

module load apps/python/conda
module load libs/cudnn/7.6.5.32/binary-cuda-10.1.243
module load apps/java/jdk1.8.0_102/binary
source activate emopia

sh run_generate_mul.sh
#sh run_train.sh
#sh run_prepare.sh
