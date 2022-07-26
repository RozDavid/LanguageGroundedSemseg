import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import logging
import torch.nn as nn
from lib.train_test.pl_BaselineTrainer import *
from lib.losses.ContrastiveLanguageLoss import ContrastiveLanguageLoss, ContrastiveLanguageCELoss


# This one is only for pretraining the model until classifier layer
class RepresentationTrainerModule(BaselineTrainerModule):

    def __init__(self, model, config, dataset):

        super().__init__(model, config, dataset)
        self.model.representation_only(True)

        self.pos_losses, self.neg_losses = AverageMeter(ignore_index=config.ignore_label), AverageMeter(ignore_index=config.ignore_label)
        self.point_to_point_losses = AverageMeter(ignore_index=config.ignore_label)
        self.feature_norm_losses = AverageMeter(ignore_index=config.ignore_label)

        if self.config.normalize_features:
            self.feat_norm_criterion = nn.MSELoss()

        self.anchor_feats = None
        self.init_criterions()

        for name, value in list(vars().items()):
            if name != "self":
                setattr(self, name, value)

    def init_criterions(self):

        self.reduction = 'none'
        # If we want to use representation loss too
        if self.config.embedding_loss_type == 'l2':
            self.embedding_criterion = nn.MSELoss(reduction=self.reduction)
        elif self.config.embedding_loss_type in ['pointsupcon', 'supcon', 'point_supcon']:
            self.embedding_criterion = PointSupConLoss(self.config, num_labels=self.num_labels, reduction=self.reduction)
            self.embedding_criterion.update_confusion_hist(torch.randint(0, self.num_labels, (self.num_labels, self.num_labels)).cuda())
        elif self.config.embedding_loss_type in ['contrast_ce']:
            self.embedding_criterion = ContrastiveLanguageCELoss(self.config, num_labels=self.num_labels, reduction=self.reduction)
        else:
            self.embedding_criterion = ContrastiveLanguageLoss(self.config, num_labels=self.num_labels, reduction=self.reduction)
            self.embedding_criterion.augment_categories = torch.nonzero(self.dataset.frequency_organized_cats[:, 2]).view(-1)  # these are the categories to be augmented in latent space

    def forward(self, x, anchor_features=None):
        if anchor_features is None:
            return self.model(x)
        else:
            return self.model(x, anchor_features)

    def on_train_epoch_start(self):
        self.reset_accumulators()

    def on_validation_epoch_start(self):
        # We have to log the training scores here due to the order of hooks
        if self.config.is_train:
            self.loop_log('training')

        # Reset accumulators
        self.reset_accumulators()

    def validation_step_end(self, outputs):

        if self.config.visualize and (self.config.visualize_freq == 0 or self.val_iteration % self.config.visualize_freq == 0 and self.val_iteration > 0):

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
                              scene_name=outputs['scene_name'],
                              output_features=outputs['output_features'])

        if self.val_iteration % self.config.test_stat_freq == 0 and self.val_iteration > 0:

            # Print info
            ious = self.iou_scores.compute() * 100.
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

        self.val_iteration += self.trainer.num_gpus

        return outputs

    def on_validation_epoch_end(self):

        # Log
        ious = self.iou_scores.compute() * 100.
        self.loop_log('validation')
        self.log("val_loss", self.losses.compute())
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

    def reset_accumulators(self):
        super().reset_accumulators()
        self.pos_losses.reset(), self.neg_losses.reset(), self.feature_norm_losses.reset(), self.point_to_point_losses.reset()

    def loop_log(self, phase='training'):

        ious = self.iou_scores.compute() * 100.

        # Write to logger
        self.log(f'{phase}/loss', self.losses.compute())
        self.log(f'{phase}/precision_at_1', self.scores.compute() * 100.)
        self.log(f'{phase}/mIoU', nanmean_t(ious))
        self.log(f'{phase}/head_mIoU', nanmean_t(ious[self.dataset.frequency_organized_cats[:, 0]]))
        self.log(f'{phase}/common_mIoU', nanmean_t(ious[self.dataset.frequency_organized_cats[:, 1]]))
        self.log(f'{phase}/tail_mIoU', nanmean_t(ious[self.dataset.frequency_organized_cats[:, 2]]))

        self.log(f'{phase}/head_loss', self.head_losses.compute())
        self.log(f'{phase}/common_loss', self.common_losses.compute())
        self.log(f'{phase}/tail_loss', self.tail_losses.compute())

        self.log(f'{phase}/head_score', nanmean_t(self.head_scores.compute()[self.dataset.frequency_organized_cats[:, 0]]) * 100.)
        self.log(f'{phase}/common_score', nanmean_t(self.common_scores.compute()[self.dataset.frequency_organized_cats[:, 0]]) * 100.)
        self.log(f'{phase}/tail_score', nanmean_t(self.tail_scores.compute()[self.dataset.frequency_organized_cats[:, 0]]) * 100.)
        self.log(f'{phase}/head_accuracy', nanmean_t(self.head_accuracy.compute()[self.dataset.frequency_organized_cats[:, 0]]) * 100.)
        self.log(f'{phase}/common_accuracy', nanmean_t(self.common_accuracy.compute()[self.dataset.frequency_organized_cats[:, 1]]) * 100.)
        self.log(f'{phase}/tail_accuracy', nanmean_t(self.tail_accuracy.compute()[self.dataset.frequency_organized_cats[:, 2]]) * 100.)

        self.log(f'{phase}/positive_loss', self.pos_losses.compute())
        self.log(f'{phase}/negative_loss', self.neg_losses.compute())

        if self.embed_losses.total > 0:
            self.log(f'{phase}/embedding_loss', self.embed_losses.compute())

        if self.feature_norm_losses.total > 0:
            self.log(f'{phase}/feature_normalization_loss', self.feature_norm_losses.compute())

        if self.point_to_point_losses.total > 0:
            self.log(f'{phase}/point_to_point_loss', self.point_to_point_losses.compute())

        if phase == 'training':
            self.log(phase + '/learning_rate', self.optimizers().param_groups[0]['lr'])

    def model_step(self, batch, batch_idx, mode='training'):

        coords, input, target, scene_name, *transform = batch
        if self.anchor_feats is None:
            self.anchor_feats = self.dataset.loaded_text_features.clone().to(target.device)

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
        if self.config.learned_projection:  # project higher dim repr space to a lower one
            sfeature_map, representation_anchors = self(*inputs, self.anchor_feats)
        else:
            sfeature_map = self(*inputs)
            representation_anchors = self.anchor_feats

        return {'feature_maps': sfeature_map,
                'target': target, 'sinput': sinput,
                'scene_name': scene_name, 'mode': mode,
                'anchor_feats': representation_anchors}


    def eval_step(self, model_step_outputs):

        soutput = model_step_outputs['feature_maps']
        target = model_step_outputs['target']
        sinput = model_step_outputs['sinput']
        scene_name = model_step_outputs['scene_name']
        anchor_feats = model_step_outputs['anchor_feats']
        mode = model_step_outputs['mode']

        # naive L2 distance loss to representations
        if self.config.embedding_loss_type == 'l2':
            # Feature clusters drove by external centroids
            loss = embedding_loss(embedding=soutput.F, target=target,
                                      feature_clusters=anchor_feats,
                                      criterion=self.embedding_criterion, config=self.config) * self.config.embedding_loss_lambda
        else:  # Contrastive
            loss, pos_loss, neg_loss = self.embedding_criterion(soutput.F, target, anchor_feats=anchor_feats)  # using all samples for negatives ass no valid prediction is learnt
            self.pos_losses(nanmean_t(pos_loss), pos_loss.size(0))
            self.neg_losses(nanmean_t(neg_loss), neg_loss.size(0))

        # We dont need attributes after this point
        if target.dim() == 2:
            target = target[:, 0]

        # Calculate split points for better logging
        valid_mask = target != self.config.ignore_label
        loss, split_losses, split_items = sample_categories_for_balancing(loss, self.config, self.dataset, targets=target, outputs=soutput.F)
        self.head_losses(nanmean_t(split_losses[0]), split_losses[0].size(0))
        self.common_losses(nanmean_t(split_losses[1]), split_losses[1].size(0))
        self.tail_losses(nanmean_t(split_losses[2]), split_losses[2].size(0))

        # Add penalty for feature norm if requested
        if self.config.normalize_features:
            feat_norm_loss = self.feat_norm_criterion(soutput.F.norm(dim=-1), torch.ones_like(target).float())
            loss += feat_norm_loss / feat_norm_loss.item() * self.config.feat_norm_loss_max  # 10% max of overall loss
            self.feature_norm_losses.update(feat_norm_loss.item(), target.shape[0])

        # Calculate cosine sim/l2 dist based preds
        feature_sims = feature_sim(soutput.F.clone().detach(), anchor_feats, config=self.config)
        pred = feature_sims.argmax(1)

        self.losses(loss.clone().detach(), target.size(0))
        self.scores(pred[valid_mask], target[valid_mask])
        self.accuracy(pred[valid_mask], target[valid_mask])
        self.iou_scores(pred[valid_mask], target[valid_mask])

        # Update scores
        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]
        self.head_scores(valid_pred[split_items[:, 0]], valid_target[split_items[:, 0]])
        self.head_accuracy(valid_pred[split_items[:, 0]], valid_target[split_items[:, 0]])
        if split_items[:, 1].sum() > 0:
            self.common_scores(valid_pred[split_items[:, 1]], valid_target[split_items[:, 1]])
            self.common_accuracy(valid_pred[split_items[:, 1]], valid_target[split_items[:, 1]])
        if split_items[:, 2].sum() > 0:
            self.tail_scores(valid_pred[split_items[:, 2]], valid_target[split_items[:, 2]])
            self.tail_accuracy(valid_pred[split_items[:, 2]], valid_target[split_items[:, 2]])

        visualize_dict = {'scene_name': scene_name, 'sinput': sinput}
        prediction_dict = {'final_pred': pred, 'final_target': target,
                           'coords': sinput.C, 'colors': sinput.F,
                           'output_features': soutput.F}
        loss_dict = {'loss': loss}

        return {**loss_dict, **visualize_dict, **prediction_dict}

