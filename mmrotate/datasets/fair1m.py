import glob
import os.path as osp
from typing import List

from mmengine.dataset import BaseDataset

from mmrotate.registry import DATASETS


@DATASETS.register_module()
class FAIR1MDataset(BaseDataset):
    """fair1m dataset for detection.

    Note: ``ann_file`` in DOTADataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTADataset,
    it is the path of a folder containing XML files.

    Args:
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
        img_suffix (str): The suffix of images. Defaults to 'png'.
    """

    METAINFO = {
        'classes':
            ('Liquid-Cargo-Ship', 'Passenger-Ship', 'Dry-Cargo-Ship', 'Cargo-Truck', 'Small-Car', 'Dump-Truck', 'Van',
             'Motorboat', 'Bridge', 'Intersection', 'Excavator', 'Engineering-Ship', 'A321', 'A220', 'other-airplane',
             'Bus', 'Tugboat', 'Tennis-Court', 'Baseball-Field', 'other-vehicle', 'Truck-Tractor', 'Fishing-Boat',
             'other-ship', 'ARJ21', 'Boeing737', 'Boeing747', 'Boeing787', 'A330', 'Football-Field', 'Basketball-Court',
             'Boeing777', 'Roundabout', 'Warship', 'C919', 'Tractor', 'A350', 'Trailer'),
        # palette is a list of color tuples, which is used for visualization.

        'palette': [(229, 22, 22), (229, 56, 22), (229, 89, 22), (229, 123, 22), (229, 156, 22), (229, 190, 22),
                    (229, 223, 22), (201, 229, 22), (168, 229, 22), (134, 229, 22), (101, 229, 22), (67, 229, 22),
                    (34, 229, 22), (22, 229, 45), (22, 229, 78), (22, 229, 112), (22, 229, 145), (22, 229, 179),
                    (22, 229, 212), (22, 212, 229), (22, 179, 229), (22, 145, 229), (22, 112, 229), (22, 78, 229),
                    (22, 45, 229), (34, 22, 229), (67, 22, 229), (101, 22, 229), (134, 22, 229), (168, 22, 229),
                    (201, 22, 229), (229, 22, 223), (229, 22, 190), (229, 22, 156), (229, 22, 123), (229, 22, 89),
                    (229, 22, 56)]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'png',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in ' f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'], img_name)

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split()
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        cls_name = bbox_info[8]
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[9])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
