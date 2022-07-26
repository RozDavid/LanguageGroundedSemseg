import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import torch
from torch import nn
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from lib.utils import nanmean_t, fast_hist_torch, per_class_iu_torch, visualize_results, print_info, loss_by_name

from lib.losses.utils import *
from lib.losses.PointSupConLoss import PointSupConLoss
from lib.losses.ContrastiveLanguageLoss import ContrastiveLanguageLoss

from lib.solvers import initialize_optimizer, initialize_scheduler
from MinkowskiEngine import SparseTensor

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from torchmetrics import Precision  # Precision@1
from torchmetrics import JaccardIndex  # IoU
from torchmetrics import AveragePrecision  # mAP
from torchmetrics import Recall  # we call it accuracy instead
from lib.losses.utils import MetricAverageMeter as AverageMeter


class BaselineTrainerModule(LightningModule):

    def __init__(self, model, config, dataset):
        super().__init__()
        for name, value in list(vars().items()):
            if name != "self":
                setattr(self, name, value)

        pl.seed_everything(config.seed)

        self.config = config
        self.model = model  # type: nn.Module
        self.save_hyperparameters(vars(config))

        self.dataset = dataset
        self.DatasetClass = load_dataset(config.dataset)
        self.num_labels = dataset.NUM_LABELS
        self.init_criterions()

        # Logging the losses
        self.losses, self.embed_losses = AverageMeter(ignore_index=config.ignore_label), AverageMeter(ignore_index=config.ignore_label)
        self.head_losses, self.common_losses, self.tail_losses = AverageMeter(ignore_index=config.ignore_label), AverageMeter(ignore_index=config.ignore_label), AverageMeter(ignore_index=config.ignore_label)

        # Logging the precisions = scores
        self.scores = Precision(num_classes=self.num_labels, average='macro')
        self.head_scores = Precision(num_classes=self.num_labels, average='none')
        self.common_scores = Precision(num_classes=self.num_labels, average='none')
        self.tail_scores = Precision(num_classes=self.num_labels, average='none')

        # Logging the accuracies = scores
        self.accuracy = Recall(num_classes=self.num_labels, average='macro')
        self.head_accuracy = Recall(num_classes=self.num_labels, average='none')
        self.common_accuracy = Recall(num_classes=self.num_labels, average='none')
        self.tail_accuracy = Recall(num_classes=self.num_labels, average='none')

        # Log IoU scores
        self.iou_scores = JaccardIndex(num_classes=self.num_labels, average='none')

        # Also tacking care of average precision - from PR curve
        self.aps = torch.zeros((0, self.num_labels))
        self.average_precision = AveragePrecision(num_classes=self.num_labels, average=None)

        # Other accumulators for logging statistics
        self.val_iteration = 0
        self.category_features = {}
        self.target_epoch_freqs = {}

    def train_dataloader(self):

        train_data_loader = initialize_data_loader(
            self.DatasetClass,
            self.config,
            phase=self.config.train_phase,
            num_workers=self.config.num_workers,
            augment_data=self.config.train_augmentation,
            shuffle=True,
            repeat=False,
            batch_size=self.config.batch_size,
            limit_numpoints=self.config.train_limit_numpoints)

        return train_data_loader

    def init_criterions(self):

        self.reduction = 'none' if self.config.balanced_category_sampling else 'mean'

        self.criterion = loss_by_name(self.config.loss_type,
                                      ignore_index=self.config.ignore_label,
                                      alpha=self.dataset.category_weights,
                                      reduction=self.reduction)

        # If we want to use representation loss too
        if self.config.embedding_loss_type == 'l2':
            self.embedding_criterion = nn.MSELoss(reduction=self.reduction)
        elif self.config.embedding_loss_type == 'contrast':
            self.embedding_criterion = ContrastiveLanguageLoss(self.config, num_labels=self.num_labels, reduction=self.reduction)
        else:
            self.embedding_criterion = PointSupConLoss(self.config, num_labels=self.num_labels)
            self.embedding_criterion.update_confusion_hist(torch.randint(0, self.num_labels, (self.num_labels, self.num_labels)).cuda())  # for hardest negative mining

        self.category_losses = torch.zeros(self.num_labels)
        self.point_frequencies = torch.zeros(self.num_labels)

    def val_dataloader(self):

        val_data_loader = initialize_data_loader(
            self.DatasetClass,
            self.config,
            num_workers=self.config.num_val_workers,
            phase=self.config.val_phase,
            augment_data=False,
            shuffle=False,
            repeat=False,
            batch_size=self.config.val_batch_size,
            limit_numpoints=False)

        self.dataset = val_data_loader.dataset
        self.init_criterions()

        self.validation_max_iter = len(val_data_loader)

        return val_data_loader

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        # necessary to always set as PL automatically switch the states on epoch end
        self.set_classifier_mode()

        # put criterions to device
        self.criterion = self.criterion.to(self.device)

        # Reset accumulators
        self.reset_accumulators()

    def on_validation_epoch_start(self):
        # We have to log the training scores here due to the order of hooks
        if self.config.is_train:
            self.loop_log('training')

        # put criterions to device
        self.criterion = self.criterion.to(self.device)

        # Reset accumulators
        self.reset_accumulators()

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        model_output = self.model_step(batch, batch_idx, mode='training')
        return self.eval_step(model_output)

    def training_step_end(self, outputs):

        if isinstance(self.embedding_criterion, PointSupConLoss):
            self.embedding_criterion.update_confusion_hist(self.iou_scores.confmat.long())

        return outputs

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        model_output = self.model_step(batch, batch_idx, mode='validation')
        return self.eval_step(model_output)

    def validation_step_end(self, outputs):

        if self.config.visualize_freq == 0 or self.val_iteration % self.config.visualize_freq == 0 and self.val_iteration > 0:

            if self.config.visualize:
                class_ids = np.arange(self.num_labels)

                label_mapper = lambda t: self.dataset.inverse_label_map[t]
                target = outputs['final_target'].cpu().apply_(label_mapper)
                pred = outputs['final_pred'].cpu().apply_(label_mapper)
                invalid_parents = target == self.config.ignore_label
                pred[invalid_parents] = self.config.ignore_label

                visualize_results(coords=outputs['coords'], colors=outputs['colors'], target=target,
                                  prediction=pred, config=self.config, iteration=self.val_iteration,
                                  num_labels=self.num_labels, train_iteration=self.global_step,
                                  valid_labels=class_ids, save_npy=True,
                                  scene_name=outputs['scene_name'])

        if self.val_iteration % self.config.test_stat_freq == 0 and self.val_iteration > 0:
            ious = self.iou_scores.compute()
            ap_class = np.nanmean(self.aps.numpy(), 0) * 100.
            class_names = self.dataset.get_classnames()
            print_info(
                self.val_iteration,
                self.validation_max_iter,
                self.losses.compute(),
                self.scores.compute() * 100.,
                ious.cpu().numpy() * 100.,
                self.iou_scores.confmat.cpu().numpy(),
                ap_class,
                class_names=class_names,
                dataset_frequency_cats=self.dataset.frequency_organized_cats)

        self.val_iteration += self.trainer.num_gpus

        return outputs

    def on_validation_epoch_end(self):

        # Log
        ious = self.iou_scores.compute() * 100.
        self.loop_log('validation')
        self.log("val_miou", nanmean_t(ious))

        # Print info
        ap_class = np.nanmean(self.aps.numpy(), 0) * 100.
        class_names = self.dataset.get_classnames()
        print_info(
            self.val_iteration,
            self.validation_max_iter,
            self.losses.compute(),
            self.scores.compute() * 100.,
            ious.cpu().numpy(),
            self.iou_scores.confmat.cpu().numpy(),
            ap_class,
            class_names=class_names,
            dataset_frequency_cats=self.dataset.frequency_organized_cats)

    def configure_optimizers(self):

        # We have to set the requires grad flags before the optimizers are initialized
        self.set_classifier_mode()

        optimizer = initialize_optimizer(self.model, self.config)
        scheduler = initialize_scheduler(optimizer, self.config)
        return [optimizer], [scheduler]

    def reset_accumulators(self):

        # Reset metric accumulators
        self.losses.reset(), self.head_losses.reset(), self.common_losses.reset(), self.tail_losses.reset()
        self.embed_losses.reset()
        self.scores.reset(), self.head_scores.reset(), self.common_scores.reset(), self.tail_scores.reset()
        self.accuracy.reset(), self.head_accuracy.reset(), self.common_accuracy.reset(), self.tail_accuracy.reset()
        self.aps = torch.zeros((0, self.num_labels))

        self.val_iteration = 0

        self.category_features = {}
        self.target_epoch_freqs = {}

        self.iou_scores.reset(), self.average_precision.reset()

    def loop_log(self, phase='training'):

        ious = self.iou_scores.compute() * 100.

        # Write to logger
        self.log(f'{phase}/loss', self.losses.compute())
        self.log(f'{phase}/precision_at_1', self.scores.compute() * 100.)
        self.log(f'{phase}/accuracy', self.accuracy.compute() * 100.)
        self.log(f'{phase}/mIoU', nanmean_t(ious))
        self.log(f'{phase}/head_mIoU', nanmean_t(ious[self.dataset.head_ids]))
        self.log(f'{phase}/common_mIoU', nanmean_t(ious[self.dataset.common_ids]))
        self.log(f'{phase}/tail_mIoU', nanmean_t(ious[self.dataset.tail_ids]))

        if self.head_losses.total > 0:
            self.log(f'{phase}/head_loss', self.head_losses.compute())
            self.log(f'{phase}/common_loss', self.common_losses.compute())
            self.log(f'{phase}/tail_loss', self.tail_losses.compute())
            self.log(f'{phase}/head_score', nanmean_t(self.head_scores.compute()[self.dataset.head_ids]) * 100.)
            self.log(f'{phase}/common_score', nanmean_t(self.common_scores.compute()[self.dataset.common_ids]) * 100.)
            self.log(f'{phase}/tail_score', nanmean_t(self.tail_scores.compute()[self.dataset.tail_ids]) * 100.)
            self.log(f'{phase}/head_accuracy', nanmean_t(self.head_accuracy.compute()[self.dataset.head_ids]) * 100.)
            self.log(f'{phase}/common_accuracy', nanmean_t(self.common_accuracy.compute()[self.dataset.common_ids]) * 100.)
            self.log(f'{phase}/tail_accuracy', nanmean_t(self.tail_accuracy.compute()[self.dataset.tail_ids]) * 100.)

        if self.embed_losses.total > 0:
            self.log(f'{phase}/embedding_loss', self.embed_losses.compute())

        if phase == 'training':
            self.log(phase + '/learning_rate', self.optimizers().param_groups[0]['lr'])

    def model_step(self, batch, batch_idx, mode='training'):

        coords, input, target, scene_name, *transform = batch

        # For some networks, making the network invariant to even, odd coords is important. Random translation
        if mode == 'training':
            coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

        # Preprocess input
        color = input[:, :3].int()
        if self.config.normalize_color:
            input[:, :3] = input[:, :3] / 255. - 0.5
        sinput = SparseTensor(input, coords)
        inputs = (sinput,) if self.config.wrapper_type == 'None' else (sinput, coords, color)
        target = target.long()

        # Feed forward
        soutput, feature_maps = self(*inputs)

        return {'soutput': soutput, 'feature_maps': feature_maps,
                'target': target, 'sinput': sinput,
                'scene_name': scene_name, 'mode': mode}

    def eval_step(self, model_step_outputs):

        soutput = model_step_outputs['soutput']
        feature_maps = model_step_outputs['feature_maps']
        target = model_step_outputs['target']
        sinput = model_step_outputs['sinput']
        scene_name = model_step_outputs['scene_name']
        mode = model_step_outputs['mode']
        valid_mask = target != self.config.ignore_label

        pred = soutput.F.max(1)[1].detach()
        prob = torch.nn.functional.softmax(soutput.F, dim=1).detach()

        # Calculate embedding and/or prediction loss
        if self.config.use_embedding_loss:
            if self.config.embedding_loss_type == 'l2':
                # Feature clusters drove by external centroids
                emb_loss = embedding_loss(embedding=feature_maps, target=target,
                                          feature_clusters=self.dataset.loaded_text_features,
                                          criterion=self.embedding_criterion, config=self.config) * self.config.embedding_loss_lambda
            else:
                # Contrastive
                if self.config.use_embedding_loss == 'both':
                    emb_loss, pos_loss, neg_loss = self.embedding_criterion(feature_maps.F, target, dataset=self.dataset, preds=pred)  # with correct predictions
                else:
                    emb_loss = self.embedding_criterion(feature_maps, target, dataset=self.dataset)  # using all samples for negatives ass no valid prediction is learnt

            self.embed_losses.update(nanmean_t(emb_loss), target.size(0))
            loss = emb_loss

            # Have both if requested
            if self.config.use_embedding_loss == 'both':
                pred_loss = self.criterion(soutput.F, target)
                if loss.shape[0] != pred_loss.shape[0]:
                    loss[target != self.config.ignore_label] += pred_loss
                else:
                    loss += pred_loss

        else:  # or only prediction if none
            loss = self.criterion(soutput.F, target)

        split_items = None
        if self.config.balanced_category_sampling:
            loss, split_losses, split_items = sample_categories_for_balancing(loss, self.config, self.dataset,
                                                                              targets=target, outputs=soutput.F)
            self.head_losses(nanmean_t(split_losses[0]), split_losses[0].size(0))
            self.common_losses(nanmean_t(split_losses[1]), split_losses[1].size(0))
            self.tail_losses(nanmean_t(split_losses[2]), split_losses[2].size(0))

        # Evaluate prediction
        if split_items is not None:
            valid_pred = pred[target != self.config.ignore_label]
            valid_target = target[target != self.config.ignore_label]
            if split_items[:, 0].sum() > 0:
                self.head_scores(valid_pred[split_items[:, 0]], valid_target[split_items[:, 0]])
                self.head_accuracy(valid_pred[split_items[:, 0]], valid_target[split_items[:, 0]])
            if split_items[:, 1].sum() > 0:
                self.common_scores(valid_pred[split_items[:, 1]], valid_target[split_items[:, 1]])
                self.common_accuracy(valid_pred[split_items[:, 1]], valid_target[split_items[:, 1]])
            if split_items[:, 2].sum() > 0:
                self.tail_scores(valid_pred[split_items[:, 2]], valid_target[split_items[:, 2]])
                self.tail_accuracy(valid_pred[split_items[:, 2]], valid_target[split_items[:, 2]])

        if valid_target.sum() > 0:
            self.losses(loss.clone().detach(), target.size(0))
            self.scores(pred[valid_mask], target[valid_mask])
            self.accuracy(pred[valid_mask], target[valid_mask])
            self.iou_scores(pred[valid_mask], target[valid_mask])

            self.aps = torch.vstack((self.aps, torch.tensor(self.average_precision(prob, target), device='cpu')))
            self.average_precision.reset()

        prediction_dict = {'final_pred': pred, 'final_target': target,
                           'coords': sinput.C, 'colors': sinput.F,
                           'output_features': prob}

        loss_dict = {'loss': loss}
        visualize_dict = {'scene_name': scene_name, 'sinput': sinput}

        return {**prediction_dict, **loss_dict, **visualize_dict}

    def test_dataloader(self):
        return self.val_dataloader()

    def on_test_start(self):
        self.target_epoch_freqs = {}
        return self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs):
        return self.validation_step_end(outputs)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
        print('===> Start testing on original pointcloud space.')
        self.dataset.test_pointcloud(self.config.visualize_path, self.num_labels)

    def set_classifier_mode(self):
        if self.config.classifier_only:
            # Freeze every layer except final
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.final.parameters():
                param.requires_grad = True