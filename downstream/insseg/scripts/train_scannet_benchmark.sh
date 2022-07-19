#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

export MODEL=Res16UNet34C
export DATAPATH='/mnt/data/Datasets/scannet_200_insseg'
export LOG_DIR='./outputs'
export PRETRAIN=''


python ddp_main.py \
    train.is_train=True \
    train.lenient_weight_loading=True \
    train.stat_freq=5 \
    train.val_freq=500 \
    net.model=${MODEL} \
    data.dataset=Scannet200Voxelization2cmDataset \
    data.batch_size=8 \
    data.num_workers=2 \
    data.scannet_path=${DATAPATH} \
    data.return_transformation=True \
    optimizer.lr=0.05 \
    optimizer.scheduler=PolyLR \
    optimizer.max_iter=20000 \
    misc.log_dir=${LOG_DIR} \
    misc.num_gpus=1 \
    hydra.launcher.comment=ECCV_supplemental \
    net.weights=$PRETRAIN \

