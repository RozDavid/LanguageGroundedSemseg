#!/bin/bash

#set -x
#set -e
#set -o pipefail


# Add project root to pythonpath
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
insseg_dir="$(dirname "$SCRIPT_DIR")"
downstream_dir="$(dirname "$insseg_dir")"
project_dir="$(dirname "$downstream_dir")"
export PYTHONPATH="${PYTHONPATH}:${project_dir}"

export BATCH_SIZE=$1
export MODEL=$2
export POSTFIX=$3
export PRETRAINED_CHECKPOINT=$4
export LOSS_TYPE=$5
export DATASET=Scannet200Voxelization2cmDataset

export DATA_ROOT="/cluster/himring/drozenberszki/ScanNet/scannet_200_insseg"
export OUTPUT_DIR_ROOT="/cluster/himring/drozenberszki/output/InstanceSegmentation"
export PRETRAINED_WEIGHTS="/cluster/himring/drozenberszki/weights/"$PRETRAINED_CHECKPOINT

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export LOG_DIR=$OUTPUT_DIR_ROOT/$DATASET/$MODEL-$POSTFIX

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR
LOG="$LOG_DIR/$TIME.txt"

python ddp_main.py \
    train.is_train=True \
    train.lenient_weight_loading=True \
    train.stat_freq=20 \
    net.model=${MODEL} \
    data.dataset=${DATASET} \
    data.batch_size=${BATCH_SIZE} \
    data.num_workers=4 \
    data.scannet_path=${DATA_ROOT} \
    data.return_transformation=True \
    optimizer.lr=0.02 \
    misc.log_dir=${LOG_DIR} \
    misc.num_gpus=8 \
    hydra.launcher.comment=arxiv_supplemental \
    optimizer.loss_type=${LOSS_TYPE} \
    net.weights=${PRETRAINED_WEIGHTS} \
    2>&1 | tee -a "$LOG"