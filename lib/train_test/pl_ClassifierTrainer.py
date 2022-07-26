import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from torch.utils.data import DataLoader
from lib.train_test.pl_BaselineTrainer import *


# This one is only for training a classifier on balanced categories
class ClassifierTrainerModule(BaselineTrainerModule):

    def __init__(self, model, config, dataset):

        super().__init__(model, config, dataset)

        self.anchor_feats = None
        self.init_criterions()

        for name, value in list(vars().items()):
            if name != "self":
                setattr(self, name, value)

    def train_dataloader(self):

        train_dataset = self.DatasetClass(self.config, phase=self.config.train_phase)

        train_data_loader = DataLoader(train_dataset,
                                       num_workers=1,
                                       shuffle=True,
                                       batch_size=self.config.batch_size,)

        self.m_train_dataloader = train_data_loader

        return train_data_loader

    def val_dataloader(self):

        val_dataset = self.DatasetClass(self.config, phase=self.config.val_phase)

        val_data_loader = DataLoader(val_dataset,
                                       num_workers=1,
                                       shuffle=True,
                                       batch_size=self.config.batch_size, )
        self.dataset = val_dataset
        self.validation_max_iter = len(val_data_loader)

        self.m_val_dataloader = val_data_loader

        return val_data_loader

    def model_step(self, batch, batch_idx, mode='training'):

        # Feed forward
        features, target = batch
        outputs, feature_maps = self(features)

        return {'feature_maps': feature_maps,
                'outputs': outputs,
                'target': target,
                'mode': mode}

    def eval_step(self, model_step_outputs):

        outputs = model_step_outputs['outputs']
        target = model_step_outputs['target']
        valid_mask = target != self.config.ignore_label

        pred = outputs.max(1)[1].detach()
        prob = torch.nn.functional.softmax(outputs, dim=1).detach()
        loss = self.criterion(outputs, target)

        head_inds, common_inds, tail_inds = self.calculate_split_items(targets=target, dataset=self.dataset)
        # Evaluate prediction
        if head_inds.sum() > 0:
            self.head_losses(nanmean_t(loss[head_inds]), head_inds.sum())
            self.head_scores(pred[head_inds], target[head_inds])
            self.head_accuracy(pred[head_inds], target[head_inds])
        if common_inds.sum() > 0:
            self.common_losses(nanmean_t(loss[common_inds]), common_inds.sum())
            self.common_scores(pred[common_inds], target[common_inds])
            self.common_accuracy(pred[common_inds], target[common_inds])
        if tail_inds.sum() > 0:
            self.tail_scores(pred[tail_inds], target[tail_inds])
            self.tail_accuracy(pred[tail_inds], target[tail_inds])
            self.tail_losses(nanmean_t(loss[tail_inds]), tail_inds.sum())

        loss = loss.mean()
        self.losses(loss.clone().detach(), target.size(0))
        self.scores(pred[valid_mask], target[valid_mask])
        self.accuracy(pred[valid_mask], target[valid_mask])
        self.iou_scores(pred[valid_mask], target[valid_mask])

        prediction_dict = {'final_pred': pred, 'final_target': target,
                           'output_features': prob}

        loss_dict = {'loss': loss}

        return {**prediction_dict, **loss_dict}

    def calculate_split_items(self, targets, dataset):

        u_values = targets.unique()

        head_inds = torch.zeros(targets.shape, dtype=bool, device=targets.device)
        common_inds = torch.zeros(targets.shape, dtype=bool, device=targets.device)
        tail_inds = torch.zeros(targets.shape, dtype=bool, device=targets.device)

        # Iterate over unique and update indexer arrays
        for unique_target in u_values:
            if unique_target.item() in dataset.head_ids:
                head_inds[targets == unique_target] = True
            elif unique_target.item() in dataset.common_ids:
                common_inds[targets == unique_target] = True
            if unique_target.item() in dataset.tail_ids:
                tail_inds[targets == unique_target] = True

        return head_inds, common_inds, tail_inds

    def on_train_epoch_end(self):
        self.m_train_dataloader.dataset.resample_features()
