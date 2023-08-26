from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptMultiConfig, OptConfigType
from mmcv.ops import DeformConv2d
import torch.nn as nn
import torch
import math
from mmrotate.models.layers.se_layer import HybridAttention
from mmcv.cnn import build_norm_layer
from mmcv.cnn import build_activation_layer
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from typing import Tuple
from torch import Tensor

"""
用于小目标的上下文信息增强(特征增强)
先使用可变形卷积，再使用空洞卷积，增加一个残差
"""


class DeformConvNet(BaseModule):
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        # 此处将out_channel设为18是因为可变形卷积操作需要18个偏移量参数 = 偏移量参数的维度2(x和y的方向) * 9(3*3的卷积核大小)
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset = nn.Conv2d(in_channels, offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(in_channels, 2 * in_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm_layer = build_norm_layer(norm_cfg, 2 * in_channels)[1]
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
                 in_channels, kernel_size=3, dilations=(1, 2, 5),
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
            out_channel = in_channels
            self.reduce_layers.append(
                ConvModule(
                    in_channels,
                    out_channel,
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
    def __init__(self, in_channel, kernel_size=(3, 5), dilation_rates=(1, 2, 5),
                 conv_cfg: bool = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = dict(type='Kaiming', layer='Conv2d',
                                                 a=math.sqrt(5), distribution='uniform',
                                                 mode='fan_in', nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        # self.deform_module1 = nn.ModuleList()
        # self.deform_module2 = nn.ModuleList()
        # self.dilation_module1 = nn.ModuleList()
        # self.dilation_module2 = nn.ModuleList()
        # self.norm_module = nn.ModuleList()
        # self.conv_module = nn.ModuleList()
        # self.attention_module = nn.ModuleList()
        # for in_channel in in_channels:
        #     self.deform_module1.append(
        #         DeformConvNet(in_channel, kernel_size=kernel_size[0],
        #                       norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        #     )
        #     self.deform_module2.append(
        #         DeformConvNet(in_channel, kernel_size=kernel_size[1],
        #                       norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        #     )
        #     self.dilation_module1.append(
        #         DilationConvNet(in_channel, kernel_size[0], dilation_rates,
        #                         conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        #     )
        #     self.dilation_module2.append(
        #         DilationConvNet(in_channel, kernel_size[1], dilation_rates,
        #                         conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg)
        #     )
        #     concat_channel = 9 * in_channel
        #     self.norm_module.append(build_norm_layer(norm_cfg, concat_channel)[1])
        #     # self.attention_module.append(HybridAttention(concat_channel, init_cfg))
        #     self.conv_module.append(ConvModule(concat_channel, in_channel, kernel_size=1, conv_cfg=conv_cfg,
        #                norm_cfg=norm_cfg, act_cfg=act_cfg))

        # 批量处理三个特征图
        # self.deform_module1 = nn.ModuleList()
        # self.deform_module2 = nn.ModuleList()
        # self.dilation_module1 = nn.ModuleList()
        # self.dilation_module2 = nn.ModuleList()
        # self.norm_module = nn.ModuleList()
        # self.conv_module = nn.ModuleList()
        # self.attention_module = nn.ModuleList()
        # self.common_module = nn.ModuleList()
        # self.depth_wise_module = nn.ModuleList()
        # for in_channel in in_channels:
        #     self.deform_module1.append(
        #         DeformDepthWiseSeparableConv(in_channel, kernel_size=kernel_size[0], norm_cfg=norm_cfg, act_cfg=act_cfg))
        #     self.deform_module2.append(
        #         DeformDepthWiseSeparableConv(in_channel, kernel_size=kernel_size[1], norm_cfg=norm_cfg, act_cfg=act_cfg))
        #     self.dilation_module1.append(
        #         DilationConvNet(in_channel, kernel_size[0], dilation_rates,
        #                         conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg))
        #     self.dilation_module2.append(
        #         DilationConvNet(in_channel, kernel_size[1], dilation_rates,
        #                         conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, init_cfg=init_cfg))
        #     concat_channel = 9 * in_channel
        #     self.norm_module.append(build_norm_layer(norm_cfg, concat_channel)[1])
        #     # self.attention_module.append(HybridAttention(concat_channel, init_cfg))
        #     self.conv_module.append(ConvModule(concat_channel, in_channel, kernel_size=1, conv_cfg=conv_cfg,
        #                norm_cfg=norm_cfg, act_cfg=act_cfg))
        #     self.common_module.append(ConvModule(in_channel, in_channel, kernel_size=1, conv_cfg=conv_cfg,
        #                                          norm_cfg=norm_cfg, act_cfg=act_cfg))

        self.deform_module1 = DeformDepthWiseSeparableConv(in_channel, kernel_size=kernel_size[0], norm_cfg=norm_cfg,
                                                           act_cfg=act_cfg)
        self.deform_module2 = DeformDepthWiseSeparableConv(in_channel, kernel_size=kernel_size[1], norm_cfg=norm_cfg,
                                                           act_cfg=act_cfg)
        self.dilation_module1 = DilationConvNet(in_channel, kernel_size[0], dilation_rates,
                                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                                                init_cfg=init_cfg)
        self.dilation_module2 = DilationConvNet(in_channel, kernel_size[1], dilation_rates,
                                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg,
                                                init_cfg=init_cfg)
        self.norm_module = build_norm_layer(norm_cfg, in_channel)[1]
        # self.attention_module.append(HybridAttention(concat_channel, init_cfg))
        self.conv_module = ConvModule(in_channel, in_channel, kernel_size=1, conv_cfg=conv_cfg,
                                      norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.activation = build_activation_layer(act_cfg)
        self.common_module = ConvModule(in_channel, in_channel, kernel_size=1, conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        # return self.concat_deform_dilation(inputs)
        return self.chain_parallel(inputs)

    def concat_deform_dilation(self, inputs):
        """
        1x1卷积并联3x3可变形卷积5x5可变形卷积并联3x3且ratio=(1,2,5)的空洞卷积并联5x5且ratio=(1,2,5)的空洞卷积
        """
        outs = []
        for idx, input in enumerate(inputs):
            def_conv1 = self.deform_module1[idx](input)
            def_conv2 = self.deform_module2[idx](input)
            dil_conv1 = self.dilation_module1[idx](input)
            dil_conv2 = self.dilation_module2[idx](input)
            res = torch.cat((input, def_conv1, def_conv2, dil_conv1, dil_conv2), dim=1)
            res = self.norm_module[idx](res)
            # res = self.attention_module[idx](res)
            res = self.conv_module[idx](res)
            outs.append(res)
        return tuple(outs)

    def chain_parallel(self, input):
        """
        1x1卷积串联3x3可变形卷积串联3x3且ratio=(1,2,5)的空洞卷积
        1x1卷积串联5x5可变形卷积串联5x5且ratio=(1,2,5)的空洞卷积
        上述两个并联输出
        """
        # outs = []
        # for idx, input in enumerate(inputs):
        #     common_conv = self.common_module[idx](input)
        #     deform_conv1 = self.deform_module1[idx](common_conv)
        #     dilation_conv1 = self.dilation_module1[idx](deform_conv1)
        #
        #     deform_conv2 = self.deform_module2[idx](input)
        #     dilation_conv2 = self.dilation_module2[idx](deform_conv2)
        #     res = dilation_conv1 + dilation_conv2 + input
        #     res = self.norm_module[idx](res)
        #     # res = self.attention_module[idx](res)
        #     res = self.conv_module[idx](res)
        #     outs.append(res)
        # return tuple(outs)
        common_conv = self.common_module(input)
        deform_conv1 = self.deform_module1(common_conv)
        dilation_conv1 = self.dilation_module1(deform_conv1)

        deform_conv2 = self.deform_module2(input)
        dilation_conv2 = self.dilation_module2(deform_conv2)
        res = dilation_conv1 + dilation_conv2 + input
        # res = self.norm_module(res)
        # res = self.attention_module[idx](res)
        # res = self.activation(res)
        return res


class DeformDepthWiseSeparableConv(BaseModule):

    def __init__(self,
                 in_channels: int,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()

        # 此处将out_channel设为18是因为可变形卷积操作需要18个偏移量参数 = 偏移量参数的维度2(x和y的方向) * 9(3*3的卷积核大小)
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset = ConvModule(in_channels, offset_channels, kernel_size=1, bias=False)
        self.depth_wise_deform = DeformConv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            dilation=dilation,
            groups=in_channels)

        self.point_wise_deform = ConvModule(
            in_channels,
            in_channels,
            kernel_size=1)

        self.norm_layer = build_norm_layer(norm_cfg, 2 * in_channels)[1]
        self.activation = build_activation_layer(act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 预测偏移量
        offset = self.conv_offset(x)
        x = self.depth_wise_deform(x, offset)
        x = self.point_wise_deform(x)
        return x
