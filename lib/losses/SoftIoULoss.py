import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes, ignore_index=255):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def to_one_hot(self, tensor, n_classes):
        n, h, w = tensor.size()
        tensor[tensor == self.ignore_index] = n_classes
        one_hot = torch.zeros(n, n_classes+1, h, w).scatter_(1, tensor.view(n, 1, h, w).cpu(), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W
        input = input[target != self.ignore_index]
        target = target[target != self.ignore_index]

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = torch.nn.functional.one_hot(target, num_classes=self.n_classes)
        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return 1 - loss.mean()
