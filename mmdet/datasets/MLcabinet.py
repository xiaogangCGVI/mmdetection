import os.path as osp

import mmcv
import numpy as np
import json
import pdb
from .custom import CustomDataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class CabinetDataset(CustomDataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = ['object']  # cautious: not contain background

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        # results['mask_fields'] = []

    def load_annotations(self, ann_file):
        img_infos = []
        with open(ann_file, 'r') as f:
            line_dict = {}
            for line in f.readlines():
                img_annotations = json.loads(line.strip())
                line_dict['filename'] = img_annotations['file']
                line_dict['width']  = img_annotations['width']
                line_dict['height']  = img_annotations['height']
                boxes_label = [(ann_dict['box'], 1) for ann_dict in img_annotations['label']] #TODO
                if boxes_label != []:
                    boxes, label = zip(*boxes_label)
                else:
                    continue
                ann = dict(bboxes=np.array(boxes).astype(np.float32),
                           labels=np.array(label).astype(np.int64))
                ann['bboxes_ignore'] = np.array([]).reshape(-1, 4).astype(np.float32) #TODO
                ann['labels_ignore'] = np.array([]).astype(np.int64)
                line_dict['ann'] = ann
                img_infos.append(line_dict)
        return img_infos

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']


