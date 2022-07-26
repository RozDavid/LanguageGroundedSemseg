# Change dataloader multiprocess start method to anything not fork
import glob
import logging
import os
import sys
import time

import numpy as np
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import random
import string

# Torch packages
import torch

# Train deps
from config.config import get_config

from lib.utils import load_state_with_same_shape, count_parameters
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset

from models import load_model, load_wrapper

import MinkowskiEngine as ME

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])

def randStr(chars = string.ascii_lowercase + string.digits, N=10):
    return ''.join(random.choice(chars) for _ in range(N))

class CleanCacheCallback(Callback):

    def training_step_end(self, trainer):
        torch.cuda.empty_cache()

    def on_train_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

    def on_validation_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

    def validation_step_end(self, trainer, pl_module):
        torch.cuda.empty_cache()


def main():
    config = get_config()

    if config.is_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")

    logging.info('===> Configurations')
    dconfig = vars(config)
    for k in dconfig:
        logging.info('    {}: {}'.format(k, dconfig[k]))

    DatasetClass = load_dataset(config.dataset)

    logging.info('===> Initializing dataloader')

    data_loader = initialize_data_loader(
        DatasetClass,
        config,
        phase=config.train_phase,
        num_workers=config.num_workers,
        augment_data=True,
        shuffle=True,
        repeat=True,
        batch_size=config.batch_size,
        limit_numpoints=config.train_limit_numpoints)

    if data_loader.dataset.NUM_IN_CHANNEL is not None:
        num_in_channel = data_loader.dataset.NUM_IN_CHANNEL
    else:
        num_in_channel = 3  # RGB color

    num_labels = data_loader.dataset.NUM_LABELS

    logging.info('===> Building model')
    NetClass = load_model(config.model)

    if config.wrapper_type == 'None':
        model = NetClass(num_in_channel, num_labels, config)
        logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__,
                                                                          count_parameters(model)))
    else:
        wrapper = load_wrapper(config.wrapper_type)
        model = wrapper(NetClass, num_in_channel, num_labels, config)

        logging.info('===> Number of trainable parameters: {}: {}'.format(
            wrapper.__name__ + NetClass.__name__, count_parameters(model)))

    # Load weights if available
    if not (config.weights == 'None' or config.weights is None):
        logging.info('===> Loading weights: ' + config.weights)
        state = torch.load(config.weights)
        if config.weights_for_inner_model:
            model.model.load_state_dict(state['state_dict'])
        else:
            if config.lenient_weight_loading:
                if 'pth' in config.weights:  # CSC version of model state
                    matched_weights = load_state_with_same_shape(model, state['state_dict'], prefix='')
                else:  # Lightning
                    matched_weights = load_state_with_same_shape(model, state['state_dict'], prefix='model.')

                model_dict = model.state_dict()
                model_dict.update(matched_weights)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(state['state_dict'])

    # Sync bathnorm for multiple GPUs
    if config.num_gpu > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    log_folder = config.log_dir
    num_devices = min(config.num_gpu, torch.cuda.device_count())
    logging.info('Starting training with {} GPUs'.format(num_devices))

    checkpoint_callbacks = [pl.callbacks.ModelCheckpoint(
        dirpath=config.log_dir,
        monitor="val_miou",
        mode='max',
        filename='checkpoint-{val_miou:.2f}-{step}',
        save_top_k=1,
        every_n_epochs=1)]

    # Set wandb project attributes
    wandb_id = randStr()
    version_num = 0
    if config.resume:
        directories = glob.glob(config.resume + '/default/*')
        versions = [int(dir.split('_')[-1]) for dir in directories]
        list_of_ckpts = glob.glob(config.resume + '/*.ckpt')

        if len(list_of_ckpts) > 0:
            version_num = max(versions) if len(versions) > 0 else 0
            ckpt_steps = np.array([int(ckpt.split('=')[1].split('.')[0]) for ckpt in list_of_ckpts])
            latest_ckpt = list_of_ckpts[np.argmax(ckpt_steps)]
            config.resume = latest_ckpt
            state_params = torch.load(config.resume)['hyper_parameters']

            if 'wandb_id' in state_params:
                wandb_id = state_params['wandb_id']
        else:
            config.resume = None
        print('Resuming: ', config.resume)
    config.wandb_id = wandb_id

    # Import the correct trainer module
    if config.use_embedding_loss and config.use_embedding_loss != 'both':
        from lib.train_test.pl_RepresentationTrainer import RepresentationTrainerModule as TrainerModule

        # we only have representation losses here
        checkpoint_callbacks += [pl.callbacks.ModelCheckpoint(
            dirpath=config.log_dir,
            monitor="val_loss",
            mode='min',
            filename='checkpoint-{val_loss:.5f}-{step}',
            save_top_k=1,
            every_n_epochs=1)]
    else:
        if 'Classifier' in config.model:
            from lib.train_test.pl_ClassifierTrainer import ClassifierTrainerModule as TrainerModule
        else:
            from lib.train_test.pl_BaselineTrainer import BaselineTrainerModule as TrainerModule

    # Init loggers
    tensorboard_logger = TensorBoardLogger(log_folder, default_hp_metric=False, log_graph=True, version=version_num)
    run_name = config.model + '-' + config.dataset if config.is_train else config.model + "_test"

    # Try a few times to avoid init error based on connection
    loggers = [tensorboard_logger]
    while config.is_train and False:
        try:
            wandb_logger = WandbLogger(project="lg_semseg", name=run_name, log_model=False, id=config.wandb_id)
            loggers += [wandb_logger]
            break
        except:
            print("Retrying WanDB connection...")
            time.sleep(10)

    trainer = Trainer(max_epochs=config.max_epoch, logger=loggers,
                      devices=num_devices, accelerator="gpu", strategy=DDPPlugin(find_unused_parameters=True),
                      num_sanity_val_steps=4, accumulate_grad_batches=1,
                      callbacks=[*checkpoint_callbacks, CleanCacheCallback()])

    pl_module = TrainerModule(model, config, data_loader.dataset)
    if config.is_train:
        trainer.fit(pl_module, ckpt_path=config.resume)
    else:
        trainer.test(pl_module, ckpt_path=config.resume)


if __name__ == '__main__':
    __spec__ = None
    main()
