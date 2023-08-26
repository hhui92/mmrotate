from mmengine.model import BaseModule
from torch import Tensor
from mmrotate.registry import MODELS
from typing import Sequence, Tuple
from mmdet.utils import ConfigType, OptMultiConfig
import math
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.layers import CSPLayer
from mmrotate.models.necks.ftm import FTM


@MODELS.register_module()
class NASCSPNeXtPAFPN(BaseModule):
    """Path Aggregation Network with CSPNeXt blocks.

    Args:
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer.
            Defaults to 3.
        use_depthwise (bool): Whether to use depthwise separable convolution in
            blocks. Defaults to False.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
            self,
            in_channels: Sequence[int],
            out_channels: int,
            num_csp_blocks: int = 3,
            use_depthwise: bool = False,
            expand_ratio: float = 0.5,
            conv_cfg: bool = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='Swish'),
            init_cfg: OptMultiConfig = dict(type='Kaiming', layer='Conv2d', a=math.sqrt(5), distribution='uniform',
                                            mode='fan_in', nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 配置文件没有传递该参数，因此使用的是ConvModule
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # 下降层 build top-down blocks
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                conv(
                    in_channels[i],
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        # 在大特征图上引出一分支使用自己设计的FTM模型提取丰富的特征信息
        self.ftm = FTM(in_channels[0])
        self.reduce_conv = ConvModule(in_channels=256, out_channels=512, kernel_size=4, stride=4)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """
        Args: inputs (tuple[Tensor]): input features.
            [128 256 256
            256 128 128
            512 64 64
            1024 32 32]
        """

        assert len(inputs) == len(self.in_channels)

        # top-down path 小图---大图进行上采样
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            if idx == 1:
                feat_low = self.ftm(inputs[idx - 1])
            else:
                feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_heigh)
            # # 增加跳连
            # if idx == 3:
            #     inner_outs[0] = feat_heigh + inputs[idx]
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(feat_heigh, scale_factor=2, mode='nearest')

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], 1))
            # 增加跳连
            if idx == 1:
                inner_out = inner_out + feat_low
            else:
                inner_out = inner_out + inputs[idx - 1]
            inner_outs.insert(0, inner_out)

        # 最上层替换为自己写的FTM模块，直接输出结果，然后将其下采样与其他层合并 经验证效果并没有很好70.1%
        # ftm = self.ftm(inputs[0])
        # inner_outs[0] = inner_outs[0] + ftm

        # outs[0]与outs[3]进行融合，缩短信息传递路径
        reduce_co = self.reduce_conv(inputs[0])
        inner_outs[-1] = reduce_co + inner_outs[-1]

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_height = inner_outs[idx + 1]
            feat_low = outs[-1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # 增加跳连
        for idx in range(len(self.in_channels)):
            outs[idx] = outs[idx] + inputs[idx]

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])
        return tuple(outs)


# in_channels = [256, 512, 1024]
# out_channels = 256
# num_csp_blocks = 3
# expand_ratio = 0.5
# # norm_cfg = dict(type='SyncBN')
# act_cfg = dict(type='SiLU')
# # t1 = torch.randn(2, 128, 256, 256)
# t2 = torch.randn(2, 256, 100, 100)
# t3 = torch.randn(2, 512, 50, 50)
# t4 = torch.randn(2, 1024, 25, 25)
# t = (t2, t3, t4)
# model = NASCSPNeXtPAFPN(in_channels, out_channels, num_csp_blocks, act_cfg=act_cfg)
# m = model(t)
# print(len(m))
