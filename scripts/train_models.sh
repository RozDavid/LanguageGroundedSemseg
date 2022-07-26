#!/bin/bash

export PYTHONUNBUFFERED="True"

export DATASET=Scannet200Voxelization2cmDataset

export MODEL=$1  #Res16UNet34C, Res16UNet34D
export BATCH_SIZE=$2
export SUFFIX=$3
export ARGS=$4

export WEIGHTS_SUFFIX=$5

export DATA_ROOT="/mnt/Data/ScanNet/scannet_200"
export PRETRAINED_WEIGHTS="/mnt/Data//weights/"$WEIGHTS_SUFFIX
export OUTPUT_DIR_ROOT="/home/drozenberszki/Downloads/tmp"

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export LOG_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL-$SUFFIX

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --scannet_path $DATA_ROOT \
    --stat_freq 100 \
    --visualize False \
    --visualize_path  $LOG_DIR/visualize \
    --num_gpu 2 \
    --balanced_category_sampling True \
    $ARGS \
    2>&1 | tee -a "$LOG"

#    --resume $LOG_DIR \
#    --weights $PRETRAINED_WEIGHTS \