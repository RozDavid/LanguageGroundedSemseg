import torch.nn as nn
import torch.nn.functional as F

class ClassifierNet(nn.Module):
    def __init__(self, num_in_channel, num_labels, config):
        super().__init__()

        self.config = config
        self.input_dim = num_in_channel
        self.output_dim = num_labels

        self.classifier = nn.Linear(self.input_dim, self.output_dim, bias=True)

    def forward(self, x):

        out = self.classifier(x)
        return out, x
