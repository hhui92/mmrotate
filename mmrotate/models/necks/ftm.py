from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptMultiConfig
from mmcv.ops import DeformConv2d
import torch.nn as nn
import torch
import math
from mmrotate.models.layers.se_layer import HybridAttention
from mmcv.cnn import build_norm_layer
from mmcv.cnn import build_activation_layer
from mmcv.cnn import ConvModule

"""
用于小目标的上下文信息增强(特征增强)
先使用可变形卷积，再使用空洞卷积，增加一个残差
"""


class DeformConvNet(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        # 此处将out_channel设为18是因为可变形卷积操作需要18个偏移量参数 = 偏移量参数的维度2(x和y的方向) * 9(3*3的卷积核大小)
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset = nn.Conv2d(in_channels, offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm_layer = build_norm_layer(norm_cfg, out_channels)[1]
        self.activation = build_activation_layer(act_cfg)

    def forward(self, x):
        # 预测偏移量
        offset = self.conv_offset(x)
        x = self.conv_adaption(x, offset)
        x = self.norm_layer(x)
        x = self.activation(x)
        return x


class DilationConvNet(BaseModule):
    def __init__(self,
                 in_channels, out_channels, kernel_size=3, dilations=(1, 2, 5),
                 conv_cfg: bool = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.dilation_tuple = dilations
        # padding保证输出图像大小与原图大小一致
        self.reduce_layers = nn.ModuleList()
        for idx in range(len(dilations)):
            p = ((kernel_size - 1) * dilations[idx]) // 2
            self.reduce_layers.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilations[idx],
                    padding=p,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            )

    def forward(self, x):
        out_list = []
        for idx in range(len(self.dilation_tuple)):
            out = self.reduce_layers[idx](x)
            out_list.append(out)
        return out_list[0] + out_list[1] + out_list[2]


class FTM(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 5), dilation_rates=(1, 2, 5),
                 conv_cfg: bool = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = dict(type='Kaiming', layer='Conv2d',
                                                 a=math.sqrt(5), distribution='uniform',
                                                 mode='fan_in', nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        self.hybrid_attention = HybridAttention(in_channels + 4 * out_channels, init_cfg)
        self.deform_conv1 = DeformConvNet(in_channels, out_channels, kernel_size=kernel_size[0],
                                          norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        self.deform_conv2 = DeformConvNet(in_channels, out_channels, kernel_size=kernel_size[1],
                                          norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        self.dilate_conv1 = DilationConvNet(in_channels, out_channels, kernel_size[0], dilation_rates,
                                            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        self.dilate_conv2 = DilationConvNet(in_channels, out_channels, kernel_size[1], dilation_rates,
                                            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        self.conv = ConvModule(in_channels + 4 * out_channels, in_channels, kernel_size=1,
                               conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        def_conv1 = self.deform_conv1(x)
        def_conv2 = self.deform_conv2(x)
        dil_conv1 = self.dilate_conv1(x)
        dil_conv2 = self.dilate_conv2(x)
        # 已保证输出尺寸与原图大小一致，可直接相拼接
        res = torch.cat((x, def_conv1, def_conv2, dil_conv1, dil_conv2), dim=1)
        res = self.hybrid_attention(res)
        res = self.conv(res)
        return res
