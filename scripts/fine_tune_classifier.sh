#!/bin/bash

export PYTHONUNBUFFERED="True"

export BATCH_SIZE=$1
export LOSS_TYPE=$2
export MODEL=$3
export ARGS=$4
export POSTFIX=$5

export DATASET=Scannet200Textual2cmDataset  # Scannet200Voxelization2cmDataset

# export DATA_ROOT="/mnt/Data/ScanNet/scannet_200"
# export LIMITED_DATA_ROOT="/mnt/Data/ScanNet/limited/"$DATASET_FOLDER
# export OUTPUT_DIR_ROOT="/mnt/Data/output"
# export PRETRAINED_WEIGHTS="/mnt/Data/weights/CLIP/Res16UNet34D.ckpt"

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL-finetune-$POSTFIX

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m lightning_main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --train_phase train \
    --scannet_path $DATA_ROOT \
    --stat_freq 40 \
    --visualize False \
    --visualize_freq 150 \
    --visualize_path  $LOG_DIR/visualize \
    --num_gpu 4 \
    --use_embedding_loss both \
    --loss_type $LOSS_TYPE \
    --classifier_only True \
    --resume $LOG_DIR \
    $ARGS \
     2>&1 | tee -a "$LOG"

#    --resume $LOG_DIR \
#    --classifier_only True \
#    --weights $PRETRAINED_WEIGHTS \