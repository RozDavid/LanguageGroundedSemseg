#!/bin/bash

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

export TEST_BATCH_SIZE=1
export MODEL=$2
export DATASET=Scannet200Voxelization2cmDataset #ScannetSuperCatVoxelization2cmDataset, Scannet4SuperCatVoxelization2cmDataset
export POSTFIX=$1

export MODEL_BASE_ROOT="/home/drozenberszki/dev/LongTailSemseg/output/models"
export OUTPUT_DIR_ROOT="/mnt/data/Outputs/LongTailSemseg/test_output"
export DATA_ROOT="/home/drozenberszki/dev/data/ScanNet/minkowski_200"
export PRETRAINED_WEIGHTS="/mnt/cluster/himring/drozenberszki/weights/CSC_ScanNet/train/Res16UNet34D.pth"

export TEST_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL$POSTFIX
export WEIGHT_DIR=$MODEL_BASE_ROOT/$DATASET/$MODEL$POSTFIX

# Save the experiment detail and dir to the common log file
mkdir -p $TEST_DIR

LOG="$TEST_DIR/$MODEL-test_log.txt"


python -m lightning_main \
    --log_dir $WEIGHT_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --is_train False \
    --val_phase val \
    --scannet_path $DATA_ROOT \
    --test_stat_freq 1 \
    --visualize_freq 0 \
    --visualize False \
    --visualize_path  $TEST_DIR/visualize \
    --sampled_features True \
    --resume $WEIGHT_DIR \
     2>&1 | tee -a "$LOG"

     #    --resume $WEIGHT_DIR \