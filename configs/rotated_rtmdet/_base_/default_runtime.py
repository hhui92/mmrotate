default_scope = 'mmrotate'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50), # 训练时日志打印间隔
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 12次保存一个模型checkpoint，最多保存3个
    checkpoint=dict(type='CheckpointHook', interval=12, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
# 加载预训练模型权重
# load_from = '/root/autodl-tmp/DIOR/trainval/work_dir/epoch_25.pth'
# load_from = None
# 是否进行模型的断点恢复
resume = False

custom_hooks = [
    dict(type='mmdet.NumClassCheckHook'),
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]
