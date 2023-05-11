import sys
sys.path.append('../')
# from oss_utils.oss_io_utils import Npz_OSS_IO
from oss_utils.text_oss_io_utils import Text_OSS_IO

import os
import torch
import numpy as np
import yaml
import random
import time

import pdb

class DummyShapes3dDataset(torch.utils.data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, num_samples, categories=None, rank=0, world_size=1):
        ''' Initialization of the the 3D shape dataset.
        Args:
            dataset_folder (str): dataset folder
            num_samples (int): num of samples in the dataset
            categories (list): list of categories to use
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.num_samples = num_samples

        use_oss = self.dataset_folder.startswith('s3://')
        self.text_oss = Text_OSS_IO(disable_client=not use_oss)
        # self.npz_oss = Npz_OSS_IO(disable_client=not use_oss)

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')
        self.metadata = self.text_oss.read(metadata_file)
        # metadata is a dict of dict, each element is like '04256520': {'id': '04256520', 'name': 'sofa,couch,lounge'}

        all_categories = [key for key in self.metadata.keys()]
        all_categories = sorted(all_categories)
        # categories is a list of categories like [04256520]

        self.all_categories = all_categories

        # Set index
        # we make sure that a category always has the same number label no matter how many categories are actually used
        for c_idx, c in enumerate(all_categories):
            self.metadata[c]['idx'] = c_idx
        # now each element in metadata is like '04256520': {'id': '04256520', 'name': 'sofa,couch,lounge', 'idx': 5}

        self.categories = self.all_categories if categories is None else categories 
        # categories is a list of categories like ['02691156', '02828884', '02933112']

        self.num_samples_per_rank = num_samples
        if world_size > 1:
            self.num_samples_per_rank = int(np.ceil(num_samples / world_size))
            if rank == world_size - 1:
                self.num_samples = num_samples - self.num_samples_per_rank * (world_size - 1)
            else:
                self.num_samples = self.num_samples_per_rank


    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return self.num_samples

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        
        category = random.choice(self.categories)
        c_idx = self.metadata[category]['idx'] # the number label of the this category
        category_name = self.metadata[category]['name'].split(',')[0]

        data = {}

        data['label'] = c_idx
        data['category'] = category
        data['category_name'] = category_name

        return data


if __name__ == '__main__':
    # dataset_folder='./shapenet_psr'
    # metadata_file = 'metadata.yaml'
    dataset_folder='s3://zylyu_datasets/shapenet_psr/'
    categories=['02958343']

    dataset = DummyShapes3dDataset(dataset_folder, num_samples=128, categories=categories)
    print(len(dataset))

    # pdb.set_trace()

    # for i in range(len(dataset)):
    #     data = dataset.__getitem__(i)
    #     print('Progress [%d|%d] % .3f' % (i, len(dataset), i/len(dataset)), data['points'].shape, data['normals'].shape, data['psr'].shape, data['label'])
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)

    for i, data in enumerate(dataloader):
        print('Progress [%d|%d] % .3f' % (i, len(dataloader), i/len(dataloader)), data['label'].shape)
    

    label = data['label'].detach().cpu().numpy()
    category = data['category']
    category_name = data['category_name']

    pdb.set_trace()

    save_file = 'temp.npz'
    np.savez(save_file, label=label, category=category, category_name=category_name)

    data2 = np.load(save_file)
    label2 = data2['label']
    category2 = data2['category'] # now it becomes a numpy array of numpy.str_
    category_name2 = data2['category_name']

    pdb.set_trace()

   
