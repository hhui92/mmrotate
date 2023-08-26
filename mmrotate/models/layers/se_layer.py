# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.utils import digit_version, is_tuple_of
from torch import Tensor
from mmrotate.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType, OptMultiConfig


@MODELS.register_module()
class SELayer(BaseModule):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Defaults to 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Defaults to (dict(type='ReLU'), dict(type='Sigmoid'))
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 channels: int,
                 ratio: int = 16,
                 conv_cfg: OptConfigType = None,
                 act_cfg: MultiConfig = (dict(type='ReLU'),
                                         dict(type='Sigmoid')),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for SELayer."""
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


@MODELS.register_module()
class DyReLU(BaseModule):
    """Dynamic ReLU (DyReLU) module.

    See `Dynamic ReLU <https://arxiv.org/abs/2003.10027>`_ for details.
    Current implementation is specialized for task-aware attention in DyHead.
    HSigmoid arguments in default act_cfg follow DyHead official code.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py

    Args:
        channels (int): The input (and output) channels of DyReLU module.
        ratio (int): Squeeze ratio in Squeeze-and-Excitation-like module,
            the intermediate channel will be ``int(channels/ratio)``.
            Defaults to 4.
        conv_cfg (None or dict): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Defaults to (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0))
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 channels: int,
                 ratio: int = 4,
                 conv_cfg: OptConfigType = None,
                 act_cfg: MultiConfig = (dict(type='ReLU'),
                                         dict(
                                             type='HSigmoid',
                                             bias=3.0,
                                             divisor=6.0)),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.channels = channels
        self.expansion = 4  # for a1, b1, a2, b2
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels * self.expansion,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        coeffs = self.global_avgpool(x)
        coeffs = self.conv1(coeffs)
        coeffs = self.conv2(coeffs) - 0.5  # value range: [-0.5, 0.5]
        a1, b1, a2, b2 = torch.split(coeffs, self.channels, dim=1)
        a1 = a1 * 2.0 + 1.0  # [-1.0, 1.0] + 1.0
        a2 = a2 * 2.0  # [-1.0, 1.0]
        out = torch.max(x * a1 + b1, x * a2 + b2)
        return out


# @MODELS.register_module()
class ChannelAttention(BaseModule):
    """Channel attention Module.
    """

    def __init__(self, channels: int, init_cfg: OptMultiConfig = None) -> None:
        super(ChannelAttention, self).__init__(init_cfg=init_cfg)
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // 16, 1, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // 16, channels, 1, bias=True)
        if digit_version(torch.__version__) < (1, 7, 0):
            self.active = nn.Hardsigmoid()
        else:
            self.active = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out
        out = self.active(out)
        return x * out


class SpatialAttention(BaseModule):
    """Spatial attention Module."""

    def __init__(self, init_cfg: OptMultiConfig = None) -> None:
        super(SpatialAttention, self).__init__(init_cfg=init_cfg)
        # 对输入特征图进行avg和max池化后再拼接，通道为2，输出时将两个通道用7x7的卷积核合为一个通道
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        if digit_version(torch.__version__) < (1, 7, 0):
            self.activate = nn.Hardsigmoid()
        else:
            self.activate = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            avg_out = torch.mean(x, dim=1, keepdim=True)
            concat_out = torch.cat([max_out, avg_out], dim=1)
        spatial_att = self.conv(concat_out)
        spatial_att = self.activate(spatial_att)
        out = x * spatial_att
        return out


# @MODELS.register_module()
class HybridAttention(BaseModule):
    """混合注意力（将通道注意力和空间注意力混合在一起）"""

    def __init__(self, channels: int, init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.channel_att = ChannelAttention(channels, init_cfg)
        self.spatial_att = SpatialAttention(init_cfg)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out
