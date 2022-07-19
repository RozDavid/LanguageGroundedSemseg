#!/bin/bash

# Exit script when a command returns nonzero state
set -x
set -e
set -o pipefail


export PYTHONUNBUFFERED="True"

export MODEL=$1  #Res16UNet34C, Res16UNet34D
export BATCH_SIZE=$2
export DATASET=Scannet200Voxelization2cmDataset
export ARGS=$3
export SUFFIX=$4
export WEIGHTS_SUFFIX=$5

export DATA_ROOT="/cluster/himring/drozenberszki/ScanNet/minkowski_200"
export PRETRAINED_WEIGHTS="/cluster/himring/drozenberszki/weights/"$WEIGHTS_SUFFIX  #"/cluster/himring/drozenberszki/weights/CLIP/ResUNet34D-CLIP-attributes.ckpt" "/cluster/himring/drozenberszki/weights/CSC_ScanNet/train/Res16UNet34D.pth"  # "/cluster/himring/drozenberszki/weights/CLIP/limited_annotation/Res16UNet34D-"$SUFFIX.ckpt    #"/cluster/himring/drozenberszki/weights/CLIP/ResUNet34D-CLIP-attributes.ckpt" "/cluster/himring/drozenberszki/weights/CLIP/ResUNet34CR-val_miou=14.79ckpt" #'/cluster/himring/drozenberszki/output/Scannet200Voxelization2cmDataset/Res16UNet34D/checkpoint-val_miou=23.59-step=227.ckpt'  "/cluster/himring/drozenberszki/weights/CSC_ScanNet/train/Res16UNet34C.pth"
export OUTPUT_DIR_ROOT="/cluster/himring/drozenberszki/output"

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export LOG_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL$SUFFIX

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m lightning_main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --train_limit_numpoints 1800000 \
    --scannet_path $DATA_ROOT \
    --stat_freq 50 \
    --visualize False \
    --visualize_freq 300 \
    --visualize_path  $LOG_DIR/visualize \
    --num_gpu 4 \
    --balanced_category_sampling True \
    --weights $PRETRAINED_WEIGHTS \
    $ARGS \
    2>&1 | tee -a "$LOG"

#    --resume $LOG_DIR \
#    --weights $PRETRAINED_WEIGHTS \