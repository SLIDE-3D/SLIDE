import sys
sys.path.append('../')
from oss_utils.oss_io_utils import Npz_OSS_IO
from oss_utils.text_oss_io_utils import Text_OSS_IO

from shapenet_psr_dataset import Shapes3dDataset
import os
import torch
import pytorch3d
from pytorch3d.ops.utils import masked_gather
import numpy as np
import pdb

if __name__ == '__main__':
    # dataset_folder='./shapenet_psr'
    metadata_file = 'metadata.yaml'
    dataset_folder='s3://zylyu_datasets/shapenet_psr/'
    num_gt_points = 2048
    split='train'
    random_subsample=False

    text_oss = Text_OSS_IO(disable_client=False)
    metadata_file = os.path.join(dataset_folder, 'metadata.yaml')
    metadata = text_oss.read(metadata_file)
    # pdb.set_trace()
    for key in metadata.keys():
        # if key == '03636649':
        category = metadata[key]['id']
        category_name = metadata[key]['name'].split(',')[0]
        dataset = Shapes3dDataset(dataset_folder, split, [category], rank=0, world_size=1)
        print(category_name, len(dataset))
    
    