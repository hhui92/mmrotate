# from mmengine.model import BaseModule
# from mmdet.utils import ConfigType, OptMultiConfig
# from mmcv.cnn import ConvModule
# import math
# from torch import Tensor
# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# from torch.autograd import Function
# from torch.nn.modules.utils import _pair
#
# """
# https://arxiv.org/pdf/2107.12292.pdf
# """
#
#
# class COT(BaseModule):
#     def __init__(self, dim, kernel_size, conv_cfg: bool = None,
#                  norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
#                  act_cfg: ConfigType = dict(type='Swish'),
#                  init_cfg: OptMultiConfig = dict(type='Kaiming', layer='Conv2d',
#                                                  a=math.sqrt(5), distribution='uniform',
#                                                  mode='fan_in', nonlinearity='leaky_relu')):
#         super().__init__(init_cfg)
#         self.dim = dim
#         self.kernel_size = kernel_size
#
#         self.key_embed = ConvModule(dim, dim, kernel_size=1, padding=kernel_size // 2, groups=4, bias=False,
#                                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
#
#         share_planes = 8
#         factor = 2
#         self.embed = nn.ModuleList()
#         self.embed.append(ConvModule(dim * 2, dim // factor, kernel_size=1, conv_cfg=conv_cfg, act_cfg=act_cfg))
#         self.embed.append(ConvModule(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1,
#                                      conv_cfg=conv_cfg, act_cfg=act_cfg))
#         self.embed.append(nn.GroupNorm(num_groups=dim // share_planes,
#                                        num_channels=pow(kernel_size, 2) * dim // share_planes))
#         self.conv1x1 = ConvModule(dim, dim, kernel_size=1, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg)
#
#         # self.conv1x1 = nn.Sequential(
#         #     nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
#         #     # nn.BatchNorm2d(dim)
#         #     ################### forget to convert BatchNorm2d get_norm(norm, dim), to fix it later
#         #     get_norm(norm, dim)
#         # )
#         # 详情看下detectron2中该部分LocalConvolution的实现
#         self.local_conv = ConvModule(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
#                                      conv_cfg=conv_cfg, act_cfg=act_cfg)
#
#         # self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
#         #                                    padding=(self.kernel_size - 1) // 2, dilation=1)
#         # self.bn = get_norm(norm, dim)
#         # act = get_act_layer('swish')
#         # self.act = act(inplace=True)
#
#         reduction_factor = 4
#         self.radix = 2
#         attn_chs = max(dim * self.radix // reduction_factor, 32)
#
#         self.se = nn.ModuleList()
#         self.se.append(ConvModule(dim, attn_chs, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg))
#         self.se.append(ConvModule(attn_chs, self.radix * dim, 1, conv_cfg=conv_cfg, act_cfg=act_cfg))
#         # self.se = nn.Sequential(
#         #     Conv2d(dim, attn_chs, 1, norm=get_norm(norm, attn_chs)),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(attn_chs, self.radix * dim, 1)
#         # )
#
#     def forward(self, x):
#         k = self.key_embed(x)
#         qk = torch.cat([x, k], dim=1)
#         b, c, qk_hh, qk_ww = qk.size()
#
#         w = self.embed(qk)
#         w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)
#
#         x = self.conv1x1(x)
#         x = self.local_conv(x, w)
#         x = self.bn(x)
#         x = self.act(x)
#
#         B, C, H, W = x.shape
#         x = x.view(B, C, 1, H, W)
#         k = k.view(B, C, 1, H, W)
#         x = torch.cat([x, k], dim=2)
#
#         x_gap = x.sum(dim=2)
#         x_gap = x_gap.mean((2, 3), keepdim=True)
#         x_attn = self.se(x_gap)
#         x_attn = x_attn.view(B, C, self.radix)
#         x_attn = F.softmax(x_attn, dim=2)
#         out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)
#
#         return out.contiguous()
#
#
# class LocalConvolution(BaseModule):
#
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int,
#         norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
#         act_cfg: ConfigType = dict(type='Swish'),
#         init_cfg: OptMultiConfig = dict(type='Kaiming', layer='Conv2d',
#                                         a=math.sqrt(5), distribution='uniform',
#                                         mode='fan_in', nonlinearity='leaky_relu')):
#         super(LocalConvolution, self).__init__(init_cfg)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#
#     def forward(self, input_tensor: Tensor, weight: Tensor):
#         assert input_tensor.shape[0] == weight.shape[0] and (input_tensor.shape[1] % weight.shape[2] == 0)
#         if input_tensor.is_cuda:
#             out = AggregationZeropad.apply(input_tensor, weight, self.kernel_size)
#         else:
#             out = AggregationZeropad.apply(input_tensor.cuda(), weight.cuda(), self.kernel_size)
#             torch.cuda.synchronize()
#             out = out.cpu()
#         return out
#
#
# class AggregationZeropad(Function):
#     pass
#     # @staticmethod
#     # def forward(ctx, input, weight, kernel_size, stride=1, padding=0, dilation=1):
#     #     kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
#     #     ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation = kernel_size, stride, padding, dilation
#     #     assert input.dim() == 4 and input.is_cuda and weight.is_cuda
#     #     batch_size, input_channels, input_height, input_width = input.size()
#     #     _, weight_heads, weight_channels, weight_kernels, weight_height, weight_width = weight.size()
#     #     output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
#     #     output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
#     #     assert output_height * output_width == weight_height * weight_width
#     #     output = input.new(batch_size, weight_heads * input_channels, output_height, output_width)
#     #     n = output.numel()
#     #     if not input.is_contiguous():
#     #         input = input.detach().clone()
#     #     if not weight.is_contiguous():
#     #         weight = weight.detach().clone()
#     #
#     #     with torch.cuda.device_of(input):
#     #         f = load_kernel('aggregation_zeropad_forward_kernel', _aggregation_zeropad_forward_kernel, Dtype=Dtype(input),
#     #                         nthreads=n,
#     #                         num=batch_size, input_channels=input_channels,
#     #                         weight_heads=weight_heads, weight_channels=weight_channels,
#     #                         bottom_height=input_height, bottom_width=input_width,
#     #                         top_height=output_height, top_width=output_width,
#     #                         kernel_h=kernel_size[0], kernel_w=kernel_size[1],
#     #                         stride_h=stride[0], stride_w=stride[1],
#     #                         dilation_h=dilation[0], dilation_w=dilation[1],
#     #                         pad_h=padding[0], pad_w=padding[1])
#     #         f(block=(CUDA_NUM_THREADS, 1, 1),
#     #           grid=(GET_BLOCKS(n), 1, 1),
#     #           args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
#     #           stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
#     #     ctx.save_for_backward(input, weight)
#     #     return output
#     #
#     # @staticmethod
#     # def backward(ctx, grad_output):
#     #     kernel_size, stride, padding, dilation = ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation
#     #     input, weight = ctx.saved_tensors
#     #     assert grad_output.is_cuda
#     #     if not grad_output.is_contiguous():
#     #         grad_output = grad_output.contiguous()
#     #     batch_size, input_channels, input_height, input_width = input.size()
#     #     _, weight_heads, weight_channels, weight_kernels, weight_height, weight_width = weight.size()
#     #     output_height, output_width = grad_output.size()[2:]
#     #     grad_input, grad_weight = None, None
#     #     opt = dict(Dtype=Dtype(grad_output),
#     #                num=batch_size, input_channels=input_channels,
#     #                weight_heads=weight_heads, weight_channels=weight_channels,
#     #                bottom_height=input_height, bottom_width=input_width,
#     #                top_height=output_height, top_width=output_width,
#     #                kernel_h=kernel_size[0], kernel_w=kernel_size[1],
#     #                stride_h=stride[0], stride_w=stride[1],
#     #                dilation_h=dilation[0], dilation_w=dilation[1],
#     #                pad_h=padding[0], pad_w=padding[1])
#     #     with torch.cuda.device_of(input):
#     #         if ctx.needs_input_grad[0]:
#     #             grad_input = input.new(input.size())
#     #             n = grad_input.numel()
#     #             opt['nthreads'] = n
#     #             f = load_kernel('aggregation_zeropad_input_backward_kernel', _aggregation_zeropad_input_backward_kernel, **opt)
#     #             f(block=(CUDA_NUM_THREADS, 1, 1),
#     #               grid=(GET_BLOCKS(n), 1, 1),
#     #               args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
#     #               stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
#     #         if ctx.needs_input_grad[1]:
#     #             grad_weight = weight.new(weight.size())
#     #             n = grad_weight.numel() // weight.shape[3]
#     #             opt['nthreads'] = n
#     #             f = load_kernel('aggregation_zeropad_weight_backward_kernel', _aggregation_zeropad_weight_backward_kernel, **opt)
#     #             f(block=(CUDA_NUM_THREADS, 1, 1),
#     #               grid=(GET_BLOCKS(n), 1, 1),
#     #               args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
#     #               stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
#     #     return grad_input, grad_weight, None, None, None, None
