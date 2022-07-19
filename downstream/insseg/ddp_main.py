# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import os
import sys
import hydra
import logging
from omegaconf import OmegaConf
import numpy as np
import torch
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.serialization import default_restore_location
import MinkowskiEngine as ME

from datasets import load_dataset
from datasets.dataset import initialize_data_loader
from insseg_models import load_model

from lib.utils import load_state_with_same_shape, count_parameters, randStr

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin, DataParallelPlugin
from pytorch_lightning import Trainer, Callback
from lib.pl_Trainer import SegmentationTrainerModule as TrainerModule

@hydra.main(config_path='config', config_name='default.yaml')
def main(config):
    # Load the configurations
    # if os.path.exists('config.yaml'):
    #     logging.info('===> Loading exsiting config file')
    #     config = OmegaConf.load('config.yaml')
    #     logging.info('===> Loaded exsiting config file')
    logging.info('===> Configurations')
    logging.info(config)

    # Dataloader
    DatasetClass = load_dataset(config.data.dataset)
    logging.info('===> Initializing dataloader')
    train_data_loader = initialize_data_loader(
        DatasetClass, config, phase=config.train.train_phase,
        num_workers=config.data.num_workers, augment_data=True,
        shuffle=True, repeat=True, batch_size=config.data.batch_size,
        limit_numpoints=config.data.train_limit_numpoints)

    # Model initialization
    logging.info('===> Building model')
    num_in_channel = train_data_loader.dataset.NUM_IN_CHANNEL
    num_labels = train_data_loader.dataset.NUM_LABELS
    NetClass = load_model(config.net.model)
    model = NetClass(num_in_channel, num_labels, config)
    logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__, count_parameters(model)))
    logging.info(model)

    # Load weights if specified by the parameter.
    if config.net.weights != '' and config.net.weights is not None:

        if not os.path.isfile(config.net.weights):
            print(f'Weight file {config.net.weights} does not exists')
        else:
            logging.info('===> Loading weights: ' + config.net.weights)
            state = torch.load(config.net.weights, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            matched_weights = load_state_with_same_shape(model, state['state_dict'])
            model_dict = model.state_dict()
            model_dict.update(matched_weights)
            model.load_state_dict(model_dict)

    # Use max GPU number
    config.misc.num_gpus = min(config.misc.num_gpus, torch.cuda.device_count())
    if config.misc.num_gpus > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint_callbacks = [
        pl.callbacks.ModelCheckpoint(dirpath=config.misc.log_dir, monitor="val_miou", mode='max', filename='checkpoint-{val_miou:.2f}-{step}', save_top_k=1, every_n_epochs=1),
        pl.callbacks.ModelCheckpoint(dirpath=config.misc.log_dir, monitor="val_map05", mode='max', filename='checkpoint-{val_map05:.2f}-{step}', save_top_k=1, every_n_epochs=1),
    ]

    # Setup Resuming
    version_num = None
    config.misc.wandb_id = randStr()
    if config.train.resume != '':
        # Remove trailing slash
        config.train.resume = config.train.resume[:-1] if config.train.resume[-1] == '/' else config.train.resume

        directories = glob.glob(config.train.resume + '/default/*')
        versions = [int(dir.split('_')[-1]) for dir in directories]
        list_of_ckpts = glob.glob(config.train.resume + '/*.ckpt')

        if len(list_of_ckpts) > 0:
            version_num = max(versions) if len(versions) > 0 else 0
            ckpt_steps = np.array([int(ckpt.split('=')[1].split('.')[0]) for ckpt in list_of_ckpts])
            latest_ckpt = list_of_ckpts[np.argmax(ckpt_steps)]
            config.train.resume = latest_ckpt
            state_params = torch.load(config.train.resume)['hyper_parameters']

            print('Resuming: ', config.train.resume)

            if 'wandb_id' in state_params and state_params['wandb_id'] != '':
                config.misc.wandb_id = state_params['wandb_id']
        else:
            config.train.resume = None
    else:
        config.train.resume = None

    # Init Loggers
    run_name = config.net.model + '-' + config.data.dataset if config.train.is_train else config.net.model + "_test"
    tensorboard_logger = TensorBoardLogger(config.misc.log_dir, default_hp_metric=False, version=version_num)
    #wandb_logger = WandbLogger(project="3DInsseg", name=run_name, log_model=False, id=config.misc.wandb_id)
    loggers = [tensorboard_logger] # , wandb_logger

    # Init PL and start
    pl_module = TrainerModule(model, config, train_data_loader.dataset)
    trainer = Trainer(max_epochs=config.optimizer.max_iter // len(train_data_loader), logger=loggers,
                      devices=config.misc.num_gpus, accelerator="gpu", strategy=DDPPlugin(find_unused_parameters=True),
                      num_sanity_val_steps=0, accumulate_grad_batches=1,
                      callbacks=[*checkpoint_callbacks])
    if config.train.is_train:
        trainer.fit(pl_module, ckpt_path=config.train.resume)
    else:
        trainer.test(pl_module, ckpt_path=config.train.resume)


if __name__ == '__main__':
    __spec__ = None
    main()
