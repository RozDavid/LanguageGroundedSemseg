#!/bin/bash

# Exit script when a command returns nonzero state
export PYTHONUNBUFFERED="True"

export BATCH_SIZE=$1
export MODEL=Res16UNet34D
export DATASET=Scannet200Textual2cmDataset

export POSTFIX=$2
export ARGS=$3

# export DATA_ROOT="/mnt/Data/ScanNet/scannet_200"
# export LIMITED_DATA_ROOT="/mnt/Data/ScanNet/limited/"$DATASET_FOLDER
# export OUTPUT_DIR_ROOT="/mnt/Data/output"
# export PRETRAINED_WEIGHTS="/mnt/Data/weights/CLIP/Res16UNet34D.ckpt"

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL-$POSTFIX

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --train_limit_numpoints 1400000 \
    --scannet_path $DATA_ROOT \
    --stat_freq 100 \
    --num_gpu 2 \
    --balanced_category_sampling False \
    --use_embedding_loss True \
    $ARGS \
    2>&1 | tee -a "$LOG"

#    --resume $LOG_DIR \
#    --weights $PRETRAINED_WEIGHTS \