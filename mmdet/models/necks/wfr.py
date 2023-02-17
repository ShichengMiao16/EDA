import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from ..builder import NECKS
from ..utils import Refine


@NECKS.register_module()
class WFR(nn.Module):
    """WFR (Weighted Fusion and Refinement module)

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        refine_level (int): Index of integration and refine level of WFR in
            multi-level features from bottom to top.
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=0):
        super(WFR, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.refine_level = refine_level
        assert 0 <= self.refine_level < self.num_levels
        self.refine = Refine(self.in_channels)

        self.convs3_list = nn.ModuleList()
        self.convs1_list = nn.ModuleList()
        for _ in range(self.num_levels):
            self.convs3_list.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    act_cfg=dict(type='ReLU')))
            self.convs1_list.append(
                nn.Conv2d(self.in_channels, 1, 1, stride=1, padding=0))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=0)

    def init_weights(self):
        """Initialize the weights of WFR module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def get_weights(self, feats_list):
        weights, output = [], []
        for i, feat in enumerate(feats_list):
            conv_feat = self.convs3_list[i](feat)
            avg_feat = self.avg_pool(conv_feat)
            weight = self.convs1_list[i](avg_feat)
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        weights = self.softmax(weights)
        for i in range(self.num_levels):
            output.append(weights[i])
        return output

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_levels

        # step 1: get weights for inputs
        weights_list = self.get_weights(inputs)

        # step 2: fuse multi-level features by resize and weighted summation
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        wff = weights_list[0] * feats[0]
        for i in range(1, self.num_levels):
            wff += weights_list[i] * feats[i]

        # step 3: refine the weighted fused features
        wff = self.refine(wff)

        # step 4: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(wff, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(wff, output_size=out_size)
            outs.append(residual + inputs[i])

        return tuple(outs)
