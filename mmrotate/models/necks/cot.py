from mmengine.model import BaseModule
from mmdet.utils import ConfigType, OptMultiConfig
from mmcv.cnn import ConvModule
import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class COT(BaseModule):
    def __init__(self, dim, kernel_size, conv_cfg: bool = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = dict(type='Kaiming', layer='Conv2d',
                                                 a=math.sqrt(5), distribution='uniform',
                                                 mode='fan_in', nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = ConvModule(dim, dim, kernel_size=1, padding=kernel_size // 2, groups=4, bias=False,
                                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        share_planes = 8
        factor = 2
        self.embed = nn.ModuleList()
        self.embed.append(ConvModule(dim * 2, dim // factor, kernel_size=1, conv_cfg=conv_cfg, act_cfg=act_cfg))
        self.embed.append(ConvModule(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1,
                                     conv_cfg=conv_cfg, act_cfg=act_cfg))
        self.embed.append(nn.GroupNorm(num_groups=dim // share_planes,
                                       num_channels=pow(kernel_size, 2) * dim // share_planes))
        self.conv1x1 = ConvModule(dim, dim, kernel_size=1, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg)

        # self.conv1x1 = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        #     # nn.BatchNorm2d(dim)
        #     ################### forget to convert BatchNorm2d get_norm(norm, dim), to fix it later
        #     get_norm(norm, dim)
        # )
        # 详情看下detectron2中该部分LocalConvolution的实现
        self.local_conv = ConvModule(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
                                     conv_cfg=conv_cfg, act_cfg=act_cfg)

        # self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
        #                                    padding=(self.kernel_size - 1) // 2, dilation=1)
        # self.bn = get_norm(norm, dim)
        # act = get_act_layer('swish')
        # self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)

        self.se = nn.ModuleList()
        self.se.append(ConvModule(dim, attn_chs, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg))
        self.se.append(ConvModule(attn_chs, self.radix * dim, 1, conv_cfg=conv_cfg, act_cfg=act_cfg))
        # self.se = nn.Sequential(
        #     Conv2d(dim, attn_chs, 1, norm=get_norm(norm, attn_chs)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(attn_chs, self.radix * dim, 1)
        # )

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)

        x = self.conv1x1(x)
        x = self.local_conv(x, w)
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)

        return out.contiguous()
