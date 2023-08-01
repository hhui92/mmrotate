# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmrotate.utils import register_all_modules


"""
python test.py --cfg-options outfile_prefix=/mnt/Dota1.0/out/offline_labelTxt
"""


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')

    parser.add_argument('--config', default='/root/mmrotate/configs/rotated_rtmdet/rotated_rtmdet_l-100e-aug-dota.py', help='test config file path')
    parser.add_argument('--checkpoint', default='/mnt/Dota1.0/trainval/work_dir/epoch_100.pth', help='checkpoint file')
    parser.add_argument('--work-dir', default='/mnt/Dota1.0/test/work_dir/',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', default='/mnt/Dota1.0/test/out/res.pkl', type=str,
        help='dump predictions to a pickle file for offline evaluation')
    # https://blog.csdn.net/qq_45708837/article/details/128383032
    # https://bbs.huaweicloud.com/blogs/324756
    # 检测整个测试集时此参数不要设置，此参数会一张一张的显示图片
    parser.add_argument('--show', action='store_true', help='show prediction results')
    # 此参数会将原图与绘制过标注信息的图合并成一张图然后保存在指定目录中
    parser.add_argument('--show-dir', default='/mnt/Dota1.0/test/show_dir/',
        help='directory where painted images will be saved.If specified, it will be automatically saved to the work_dir/timestamp/show_dir')

    """并生成txt和png文件提交给官方评估服务器
        python test.py configs/rotated_rtmdet/rotated_rtmdet_s-3x-dota.py
        --format-only  --cfg-options outfile_prefix=/mnt/Dota1.0/out/offline_labelTxt"
        生成的png和txt将在./mask_rcnn_cityscapes_test_results目录下"""
    # 此处内容在configs/rotated_rtmdet/_base_/dota_rr.py文件中已经设置，
    # 当换数据集试验时再设置此处内容
    # parser.add_argument('--format-only', action='store_true',
    #     default=True,
    #     help='Format the output results without perform evaluation. It is'
    #     'useful when you want to format the result to a specific format and '
    #     'submit it to the test server')

    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage "visualization=dict(type=\'VisualizationHook\')"')

    return cfg


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

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
