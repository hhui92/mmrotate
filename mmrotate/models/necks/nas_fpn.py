from mmengine.model import BaseModule
from mmrotate.registry import MODELS
from typing import Sequence, Tuple
from mmdet.utils import ConfigType, OptMultiConfig
import math
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torch
import torch.nn as nn
from ..layers import CSPLayer



@MODELS.register_module()
class NASFPN(BaseModule):
    """
    将CSPNeXtPAFPN的CSPLayer的卷积核换成空洞卷积试一下效果
    实现DCFPN看下效果
    用NAS搜索DCFPN和CSPNeXtPAFPN试下效果
    """

    def __init__(
            self,
            in_channels: Sequence[int],
            out_channels: int,
            num_csp_blocks: int = 3,
            use_depthwise: bool = False,
            expand_ratio: float = 0.5,
            upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
            conv_cfg: bool = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='Swish'),
            ## kaiming均匀初始化方法具体作用查一查
            init_cfg: OptMultiConfig = dict(
                type='Kaiming',
                layer='Conv2d',
                a=math.sqrt(5),
                distribution='uniform',
                mode='fan_in',
                nonlinearity='leaky_relu')
    ) -> None:
        super(NASFPN, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
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
















        super(NASFPN, self).__init__()
        # 定义六层卷积层
        # 两层HDC（1,2,5,1,2,5）
        self.conv = nn.Sequential(
            # 第一层 (3-1)*1+1=3 （64-3)/1 + 1 =62
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(32),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第二层 (3-1)*2+1=5 （62-5)/1 + 1 =58
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=2),
            nn.BatchNorm2d(32),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第三层 (3-1)*5+1=11  (58-11)/1 +1=48
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=5),
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第四层(3-1)*1+1=3 （48-3)/1 + 1 =46
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第五层 (3-1)*2+1=5 （46-5)/1 + 1 =42
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=2),
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第六层 (3-1)*5+1=11  (42-11)/1 +1=32
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=5),
            nn.BatchNorm2d(128),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True)
        )
        # 输出层,将通道数变为分类数量
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # 图片经过三层卷积，输出维度变为(batch_size,C_out,H,W)
        out = self.conv(x)
        # 使用平均池化层将图片的大小变为1x1,第二个参数为最后输出的长和宽（这里默认相等了）
        out = F.avg_pool2d(out, 32)
        # 将张量out从shape batchx128x1x1 变为 batch x128
        out = out.squeeze()
        # 输入到全连接层将输出的维度变为3
        out = self.fc(out)
        return out
