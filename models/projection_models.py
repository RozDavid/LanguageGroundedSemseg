import torch
import torch.nn as nn

class AttributeFittingModel(nn.Module):
    def __init__(self, inputSize, outputSize, num_attributes):
        super(AttributeFittingModel, self).__init__()

        self.input_size = inputSize
        self.output_size = outputSize
        self.num_attributes = num_attributes

        self.attr_linears = nn.ModuleList([nn.Linear(inputSize, outputSize) for i in range(num_attributes)])

    def forward(self, x):
        # x = num_cats x num_attrs x num_dims
        out = torch.cuda.FloatTensor(x.shape[0], self.num_attributes, self.output_size).fill_(0)  # type: torch.FloatTensor
        for i in range(len(self.attr_linears)):
            out[:, i, :] = self.attr_linears[i](x)

        return out