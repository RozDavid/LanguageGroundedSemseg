import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

from models.res16unet import *
from models.modules.resnet_block import NoReluBlock

class Res16UNet34GloVe(Res16UNet34C100):

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34GloVe, self).__init__(in_channels, out_channels, config, D)


    def forward(self, x):

        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # 1/8 resolution -> 1/16
        # stride 2, dilation 1
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # 1/16 resolution -> 1/8
        # up_stride 2, dilation 1
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/8
        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = me.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out), out.F


# Representation out put and possibly final output with baseline dimensionality
class Res16UNet34CR(Res16UNet34C):

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34CR, self).__init__(in_channels, out_channels, config, D)

        # We don't need relu in the end
        self.block8[-1] = NoReluBlock(self.block8[-1])

        # For autograd
        self.repr_only = False

    def representation_only(self, flag):
        # For autograd
        self.repr_only = flag
        self.final = None

    def forward(self, x):  # remove final classifier layer
        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # 1/8 resolution -> 1/16
        # stride 2, dilation 1
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # 1/16 resolution -> 1/8
        # up_stride 2, dilation 1
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/8
        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = me.cat(out, out_p1)
        out = self.block8(out)

        if self.repr_only:
            return out
        else:
            return self.final(out), out


# Learn the projection function from a higher dimensional representation space to a smaller one
class Res16UNet34CR_Proj(Res16UNet34CR):

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34CR_Proj, self).__init__(in_channels, out_channels, config, D)

        self.projection_layer = nn.Conv1d(512, self.PLANES[7], kernel_size=1, stride=1, bias=True)  # TODO remove hardcoded dim ugh

    def forward(self, x, achor_feats):
        return super().forward(x), self.projection_layer(achor_feats.unsqueeze(-1)).squeeze()



# Representation out put and possibly final output with CLIP dimensionality
class Res16UNet34D(Res16UNet34CR):  # For CLIP
    PLANES = (32, 64, 128, 256, 256, 256, 256, 512)

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34D, self).__init__(in_channels, out_channels, config, D)

        # We don't need relu in the end
        self.block8[-1] = NoReluBlock(self.block8[-1])

        # For autograd
        self.repr_only = False


# SimSiam-like model with shared backbones
class Res16UNet34DPaired(Res16UNet34): # For SimSiam training
    PLANES = (32, 64, 128, 256, 256, 256, 256, 512)  # CLIP model
    # PLANES = (32, 64, 128, 256, 256, 128, 96, 96)  # baseline model

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34DPaired, self).__init__(in_channels, out_channels, config, D)

        # have output on same dimension
        # self.final = conv(self.PLANES[7], self.PLANES[7], kernel_size=1, stride=1, bias=True, D=D)

        # For autograd
        self.repr_only = False

    def representation_only(self, flag):
        # For autograd
        self.repr_only = flag
        self.final = None

    def base_forward(self, x):

        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # 1/8 resolution -> 1/16
        # stride 2, dilation 1
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # 1/16 resolution -> 1/8
        # up_stride 2, dilation 1
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/8
        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return out  #self.final(out),

    def forward(self, x1, x2):  # take dual inputs for paired matching

        z1 = self.base_forward(x1)
        z2 = self.base_forward(x2)

        return z1.F, z2.F

# Outdated
class Res16UNet34C_P(Res16UNet34C):

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34C_P, self).__init__(in_channels, out_channels, config, D)

        # We don't need relu in the end
        self.block8[-1] = NoReluBlock(self.block8[-1])

    def forward(self, x):  # remove final classifier layer
        # Input resolution stride, dilation 1
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        # Input resolution -> half
        # stride 2, dilation 1
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        # Half resolution -> quarter
        # stride 2, dilation 1
        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        # 1/4 resolution -> 1/8
        # stride 2, dilation 1
        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # 1/8 resolution -> 1/16
        # stride 2, dilation 1
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # 1/16 resolution -> 1/8
        # up_stride 2, dilation 1
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        # Concat with res 1/8
        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # 1/8 resolution -> 1/4
        # up_stride 2, dilation 1
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        # Concat with res 1/4
        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # 1/4 resolution -> 1/2
        # up_stride 2, dilation 1
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        # Concat with res 1/2
        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # 1/2 resolution -> orig
        # up_stride 2, dilation 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # Concat with orig
        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out), out


# Train only classifier with more complex final layer
class Res16UNet34Dv2(Res16UNet34D):  # For CLIP frozen weights

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34Dv2, self).__init__(in_channels, out_channels, config, D)

        self.final = nn.Sequential(
            conv(self.PLANES[7], 512, kernel_size=1, bias=True, D=D),
            conv(512, 512, kernel_size=1, bias=True, D=D),
            ME.MinkowskiInstanceNorm(512),
            ME.MinkowskiReLU(),
            conv(512, out_channels, kernel_size=1, bias=True, D=D))


# Train only classifier with more complex final layer
class Res16UNet34Dv3(Res16UNet34D):  # For CLIP frozen weights

    def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
        super(Res16UNet34Dv3, self).__init__(in_channels, out_channels, config, D)

        bn_momentum = config.bn_momentum

        self.final = nn.Sequential(
            self._make_layer(self.BLOCK, self.inplanes, 1, dilation=1, norm_type=NormType.INSTANCE_NORM, bn_momentum=bn_momentum),
            ME.MinkowskiInstanceNorm(512),
            ME.MinkowskiReLU(),
            conv(self.PLANES[7], 512, kernel_size=1, bias=True, D=D),
            conv(512, 512, kernel_size=1, bias=True, D=D),
            ME.MinkowskiInstanceNorm(512),
            ME.MinkowskiReLU(),
            conv(512, out_channels, kernel_size=1, bias=True, D=D))
