import argparse
import logging
import os
import os.path as osp
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmrotate.utils import register_all_modules


"""
训练教程 http://www.xbhp.cn/news/28200.html
nohup python train.py > /root/autodl-tmp/DOTA_ms/runlog/runlog.log 2>&1 & 
nohup python train.py > /root/autodl-tmp/DIOR/runlog/runlog.log 2>&1 & 
nohup python train.py > /root/autodl-tmp/FAIR1M_ms/runlog/runlog.log 2>&1 & 
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    # parser.add_argument('--config', default='/root/mmrotate/configs/rotated_rtmdet/rotated_rtmdet_l-3x-dota_ms.py', help='train config file path')
    # parser.add_argument('--work-dir', default='/root/autodl-tmp/DOTA-v1.0/trainval/work_dir/', help='the dir to save logs and models')

    parser.add_argument('--config', default='/root/mmrotate/configs/rotated_rtmdet/rotated_rtmdet_l-100e-aug-dior.py', help='train config file path')
    parser.add_argument('--work-dir', default='/root/autodl-tmp/DIOR/trainval/work_dir/', help='the dir to save logs and models')

    # parser.add_argument('--config', default='/root/mmrotate/configs/rotated_rtmdet/rotated_rtmdet_l-3x-fair1m_ms.py', help='train config file path')
    # parser.add_argument('--work-dir', default='/root/autodl-tmp/FAIR1M_ms/train/work_dir/', help='the dir to save logs and models')

    parser.add_argument('--seed', default=42)
    # parser.add_argument('--deterministic', default=True)
    parser.add_argument('--amp', action='store_true', default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument('--auto-scale-lr', action='store_true', help='enable automatically scaling LR.')
    # action='store_true'的默认值为False, 要想继续之前的训练需要将default设为True,此处的参数优先级最高，会覆盖掉配置文件中的参数值
    parser.add_argument('--resume', action='store_true', default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
