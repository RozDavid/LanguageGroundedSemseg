import warnings

from downstream.insseg.lib.utils import visualize_instances

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import numpy as np
import torch

from torch import nn
from MinkowskiEngine import SparseTensor

from lib.solvers import initialize_optimizer, initialize_scheduler
from datasets import load_dataset
from datasets.dataset import initialize_data_loader
from lib.utils import precision_at_one, get_prediction, print_info, nanmean_t
from lib.losses.utils import loss_by_name

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from torchmetrics import JaccardIndex  # IoU
from torchmetrics import AveragePrecision  # mAP
from torchmetrics import Precision  # Precision@1
from torchmetrics import Recall  # Accuracy

from lib.bfs.bfs import Clustering
from datasets.evaluation.evaluate_semantic_instance import Evaluator as InstanceEvaluator
from lib.utils import MetricAverageMeter as AverageMeter

class SegmentationTrainerModule(LightningModule):

    def __init__(self, model, config, dataset):
        super().__init__()
        for name, value in list(vars().items()):
            if name != "self":
                setattr(self, name, value)

        # fix seed
        pl.seed_everything(config.misc.seed)

        self.config = config
        self.model = model  # type: nn.Module
        self.save_hyperparameters(config)

        self.dataset = dataset
        self.DatasetClass = load_dataset(config.data.dataset)
        self.num_labels = dataset.NUM_LABELS
        self.ignore_label = self.config.data.ignore_label
        self.init_criterions()

        # Init accumulators
        self.scores = Precision(num_classes=self.num_labels, average='macro')
        self.accuracy = Recall(num_classes=self.num_labels, average='macro')

        self.semantic_loss = AverageMeter()
        self.offset_dir_loss = AverageMeter()
        self.offset_norm_loss = AverageMeter()
        self.total_loss = AverageMeter()

        self.iou_scores = JaccardIndex(num_classes=self.num_labels, reduction='none')

        # Also tacking care of average precision - from PR curve
        self.aps = torch.zeros((0, self.num_labels))
        self.average_precision = AveragePrecision(num_classes=self.num_labels, average=None)

        self.val_iteration = 0


    def init_criterions(self):

        self.reduction = 'mean'
        self.criterion = loss_by_name(self.config.optimizer.loss_type,
                                      ignore_index=self.ignore_label,
                                      alpha=self.dataset.category_weights,
                                      reduction=self.reduction)

    def configure_optimizers(self):

        optimizer = initialize_optimizer(self.model.parameters(), self.config.optimizer)
        scheduler = initialize_scheduler(optimizer, self.config.optimizer)

        return [optimizer], [scheduler]

    def train_dataloader(self):

        train_data_loader = initialize_data_loader(
            self.DatasetClass, self.config, phase=self.config.train.train_phase,
            num_workers=self.config.data.num_workers, augment_data=True,
            shuffle=True, repeat=False, batch_size=self.config.data.batch_size // self.config.misc.num_gpus,
            limit_numpoints=self.config.data.train_limit_numpoints)

        return train_data_loader

    def val_dataloader(self):

        ######################################################################################
        #  Added for Instance Segmentation
        ######################################################################################
        self.VALID_CLASS_IDS = torch.FloatTensor(self.dataset.VALID_CLASS_IDS).long()
        self.CLASS_LABELS_INSTANCE = self.dataset.CLASS_LABELS if self.config.misc.train_stuff else self.dataset.CLASS_LABELS_INSTANCE
        self.VALID_CLASS_IDS_INSTANCE = self.dataset.VALID_CLASS_IDS if self.config.misc.train_stuff else self.dataset.VALID_CLASS_IDS_INSTANCE
        self.IGNORE_LABELS_INSTANCE = self.dataset.IGNORE_LABELS if self.config.misc.train_stuff else self.dataset.IGNORE_LABELS_INSTANCE
        self.evaluator = InstanceEvaluator(self.CLASS_LABELS_INSTANCE, self.VALID_CLASS_IDS_INSTANCE)

        self.cluster_thresh = 1.5
        self.propose_points = 100
        self.score_func = torch.mean

        self.cluster = Clustering(ignored_labels=self.IGNORE_LABELS_INSTANCE,
                             class_mapping=self.VALID_CLASS_IDS,
                             thresh=self.cluster_thresh,
                             score_func=self.score_func,
                             propose_points=self.propose_points,
                             closed_points=300,
                             min_points=20)


        if self.config.test.dual_set_cluster:
            # dual set clustering when submit to benchmark
            self.cluster_ = Clustering(ignored_labels=self.IGNORE_LABELS_INSTANCE,
                                  class_mapping=self.VALID_CLASS_IDS,
                                  thresh=0.05,
                                  score_func=torch.mean,
                                  propose_points=250,
                                  closed_points=300,
                                  min_points=50)

        val_data_loader = initialize_data_loader(
            self.DatasetClass, self.config, phase=self.config.train.val_phase,
            num_workers=4, augment_data=False,
            shuffle=True, repeat=False,
            batch_size=1, limit_numpoints=False)
        self.val_dataset = val_data_loader.dataset

        self.validation_max_iter = len(val_data_loader)

        return val_data_loader

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        # Reset accumulators
        self.reset_accumulators()

    def on_validation_epoch_start(self):
        # We have to log the training scores here due to the order of hooks
        if self.config.train.is_train:
            self.loop_log('training')

        # Reset accumulators
        self.reset_accumulators()

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        model_output = self.model_step(batch, batch_idx, mode='training')
        return model_output

    def training_step_end(self, outputs):

        if self.global_step % self.config.train.stat_freq == 0 or self.global_step == 1:
            self.loop_log()
            self.reset_accumulators()

        return outputs

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        model_output = self.model_step(batch, batch_idx, mode='validation')
        return self.eval_step(model_output)

    def validation_step_end(self, outputs):

        if self.val_iteration % self.config.test.test_stat_freq == 0 and self.val_iteration > 0:
            ious = self.iou_scores.compute()
            ap_class = np.nanmean(self.aps.numpy(), 0) * 100.
            class_names = self.dataset.get_classnames()
            print_info(
                self.val_iteration,
                self.validation_max_iter,
                self.total_loss.compute(),
                self.scores.compute() * 100.,
                ious.cpu().numpy() * 100.,
                self.iou_scores.confmat.cpu().numpy(),
                ap_class,
                class_names=class_names)

        self.val_iteration += self.trainer.num_gpus

        if self.config.test.visualize:
            visualize_instances(coords=outputs['soutput'].C, clustered_results=outputs['clustered_result'],
                                labels=outputs['preds'], gt=outputs['target'], gt_instances=outputs['instances'][0],
                                config=self.config, scene_name=outputs['scene_names'][0])

        return outputs

    def on_validation_epoch_end(self):

        # Calculate mAP@0.5
        _, mAP50, _ = self.evaluator.evaluate()

        # Log
        self.loop_log('validation', val_mAP=mAP50)
        self.log('val_miou', nanmean_t(self.iou_scores.compute()) * 100.)
        self.log('val_map05', mAP50 * 100.)

        # Print info
        ap_class = np.nanmean(self.aps.numpy(), 0) * 100.
        class_names = self.dataset.get_classnames()
        print_info(
            self.val_iteration,
            self.validation_max_iter,
            self.total_loss.compute(),
            self.scores.compute() * 100.,
            self.iou_scores.compute().cpu().numpy() * 100.,
            self.iou_scores.confmat.cpu().numpy(),
            ap_class,
            class_names=class_names)

    def reset_accumulators(self):

        self.semantic_loss.reset(), self.total_loss.reset(), self.offset_norm_loss.reset(), self.offset_dir_loss.reset()

        self.scores.reset()
        self.accuracy.reset()
        self.iou_scores.reset()
        self.val_iteration = 0

    def loop_log(self, phase='training', val_mAP=None):
        self.log(f'{phase}/miou', nanmean_t(self.iou_scores.compute()) * 100.)
        self.log(f'{phase}/precision_at_1', nanmean_t(self.scores.compute()) * 100.)
        self.log(f'{phase}/accuracy', nanmean_t(self.accuracy.compute()) * 100.)
        self.log(f'{phase}/total_loss', self.total_loss.compute())
        self.log(f'{phase}/semantic_loss', self.semantic_loss.compute())
        self.log(f'{phase}/offset_dir_loss', self.offset_dir_loss.compute())
        self.log(f'{phase}/offset_norm_loss', self.offset_norm_loss.compute())

        # Learning rate
        if phase == 'training':
            self.log(f'{phase}/learning_rate', self.optimizers().param_groups[0]['lr'])


    def model_step(self, batch, batch_idx, mode='training'):

        batch_score = 0
        batch_losses = {
            'semantic_loss': 0.0,
            'offset_dir_loss': 0.0,
            'offset_norm_loss': 0.0,
            'total_loss': 0.0}

        if self.config.data.return_transformation:
            coords, input, target, instances, scene_names, transformation = batch
        else:
            coords, input, target, instances, scene_names = batch
            transformation = None

        # Preprocess input
        if self.config.augmentation.normalize_color:
            input[:, :3] = input[:, :3] / 255. - 0.5
        sinput = SparseTensor(input, coords, device=target.device)
        inputs = (sinput,)
        target = target.long()

        # Feed forward
        pt_offsets, soutput, out_feats = self(*inputs)
        output = soutput.F

        # -----------------semantic loss----------------------
        semantic_loss = self.criterion(soutput.F, target.long())
        total_loss = semantic_loss

        # -----------------offset loss----------------------
        ## pt_offsets: (N, 3), float, cuda
        ## coords: (N, 3), float32
        ## centers: (N, 3), float32 tensor
        ## instance_ids: (N), long
        centers = np.concatenate([instance['center'] for instance in instances])
        instance_ids = np.concatenate([instance['ids'] for instance in instances])

        centers = torch.from_numpy(centers).cuda()
        instance_ids = torch.from_numpy(instance_ids).cuda().long()

        gt_offsets = centers - coords[:, 1:].cuda()  # (N, 3)
        gt_offsets *= self.dataset.VOXEL_SIZE
        pt_diff = pt_offsets.F - gt_offsets  # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # (N)
        valid = (instance_ids != -1).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets.F, p=2, dim=1)
        pt_offsets_ = pt_offsets.F / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)
        total_loss += offset_norm_loss + offset_dir_loss

        # Compute preds
        pred = get_prediction(self.dataset, output, target)
        prob = torch.nn.functional.softmax(output, dim=1)

        self.total_loss(total_loss.clone().detach(), target.size(0))
        self.semantic_loss(semantic_loss.clone().detach(), target.size(0))
        self.offset_dir_loss(offset_dir_loss.clone().detach(), target.size(0))
        self.offset_norm_loss(offset_norm_loss.clone().detach(), target.size(0))

        # Update metrics
        valid_mask = target != self.ignore_label
        self.scores(pred[valid_mask], target[valid_mask])
        self.accuracy(pred[valid_mask], target[valid_mask])
        self.iou_scores(pred[valid_mask], target[valid_mask])

        self.aps = torch.vstack((self.aps, torch.tensor(self.average_precision(prob, target))))
        self.average_precision.reset()

        return {'soutput': soutput, 'target': target, 'sinput': sinput, 'scene_names': scene_names, 'loss': total_loss,
                'mode': mode, 'batch_losses': batch_losses, 'batch_score': batch_score, 'preds': pred,
                'transformation': transformation, 'pt_offsets': pt_offsets, 'instances': instances}

    def eval_step(self, model_step_outputs):

        coords = model_step_outputs['sinput'].C
        output = model_step_outputs['soutput'].F
        transformation = model_step_outputs['transformation']
        pt_offsets = model_step_outputs['pt_offsets']
        target = model_step_outputs['target']
        instances = model_step_outputs['instances']

        #####################################################################################
        #  Added for Instance Segmentation
        ######################################################################################
        if self.config.test.evaluate_benchmark:
            # ---------------- point level -------------------
            # voting loss for dual set clustering, w/o using ScoreNet
            scene_id = self.val_dataset.get_output_id(self.val_iteration)
            inverse_mapping = self.val_dataset.get_original_pointcloud(coords.cpu(), transformation.cpu(), self.val_iteration)
            vertices = inverse_mapping[1] + pt_offsets.F[inverse_mapping[0]].cpu().numpy()
            features = output[inverse_mapping[0]]
            instances = self.cluster.get_instances(vertices, features)
            if self.config.test.dual_set_cluster:
                instances_ = self.cluster_.get_instances(inverse_mapping[1], features)
                instances = self.nms(instances, instances_)
            self.evaluator.add_prediction(instances, scene_id)
            # comment out when evaluate on benchmark format
            self.evaluator.add_gt_in_benchmark_format(scene_id)
            self.evaluator.write_to_benchmark(scene_id=scene_id, pred_inst=instances)

            # For voxel level
            vertices = coords.cpu().numpy()[:, 1:] + pt_offsets.F.cpu().numpy() / self.dataset.VOXEL_SIZE
            clustered_result = self.cluster.get_instances(vertices, output.clone().cpu())
        else:
            # --------------- voxel level------------------
            vertices = coords.cpu().numpy()[:, 1:] + pt_offsets.F.cpu().numpy() / self.dataset.VOXEL_SIZE
            clustered_result = self.cluster.get_instances(vertices, output.clone().cpu())
            instance_ids = instances[0]['ids']
            gt_labels = target.clone()
            gt_labels[instance_ids == -1] = self.IGNORE_LABELS_INSTANCE[0]  # invalid instance id is -1, map 0,1,255 labels to 0
            gt_labels = self.VALID_CLASS_IDS[gt_labels.long()]
            self.evaluator.add_gt((gt_labels * 1000 + instance_ids).numpy(), self.val_iteration)  # map invalid to invalid label, which is ignored anyway
            self.evaluator.add_prediction(clustered_result, self.val_iteration)
        ######################################################################################

        model_step_outputs['clustered_result'] = clustered_result

        return model_step_outputs

    def nms(self, instances, instances_):
        instances_return = {}
        counter = 0
        for key in instances:
            label = instances[key]['label_id'].item()
            if label in [10, 12, 16]:
                continue
            instances_return[counter] = instances[key]
            counter += 1

        # dual set clustering, for some classes, w/o voting loss is better
        for key_ in instances_:
            label_ = instances_[key_]['label_id'].item()
            if label_ in [10, 12, 16]:
                instances_return[counter] = instances_[key_]
                counter += 1

        return instances_return


    def test_dataloader(self):
        return self.val_dataloader()

    def on_test_start(self):
        return self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        return self.validation_step(batch, batch_idx)

    def test_step_end(self, outputs):
        return self.validation_step_end(outputs)

    def on_test_epoch_end(self):

        # Calculate mAP@0.5
        _, mAP50, _ = self.evaluator.evaluate()

        # Print info
        ap_class = np.nanmean(self.aps.numpy(), 0) * 100.
        class_names = self.dataset.get_classnames()
        print_info(
            self.val_iteration,
            self.validation_max_iter,
            self.total_loss.compute(),
            self.scores.compute() * 100.,
            self.iou_scores.compute().cpu().numpy() * 100.,
            self.iou_scores.confmat.cpu().numpy(),
            ap_class,
            class_names=class_names)

