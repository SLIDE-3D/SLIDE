import sys
sys.path.append('../')
from oss_utils.oss_io_utils import Npz_OSS_IO
from oss_utils.text_oss_io_utils import Text_OSS_IO

import os
import logging
# from torch.utils import data
import torch
import numpy as np
import yaml
import random
import time
import copy

import pdb

class Shapes3dDataset(torch.utils.data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, split=None, categories=None, scale=1, num_gt_points=2048, rank=0, world_size=1,
                append_samples_to_last_rank=True, shuffle_before_rank_split=True, load_psr=False, augmentation=False,
                random_subsample=False, num_samples=1000, repeat_dataset=1, centered_to_centroid=True):
        ''' Initialization of the the 3D shape dataset.
        Args:
            dataset_folder (str): dataset folder
            split (str): which split is used, should be train, test or val
            categories (list): list of categories to use
            scale of the point cloud
            number of points to subsample
            repeat_dataset (int): we may want to repeat the models in the dataset when there are two few shapes in the dataset, we avoid frequently reload the dataset
        '''
        assert split in [None, 'train', 'val', 'test']
        if repeat_dataset > 1:
            assert split == 'train'
            if random_subsample:
                print('will not repeat the dataset because need to random_subsample')
                repeat_dataset = 1
        # Attributes
        self.dataset_folder = dataset_folder
        self.num_gt_points = num_gt_points
        self.scale = scale
        self.load_psr = load_psr
        # self.noise_magnitude = noise_magnitude # this gaussian noise is added to both points and normals
        self.augmentation = augmentation # augmentation could be a dict or False
        print('dataset augmentation is', self.augmentation, flush=True)
        self.centered_to_centroid = centered_to_centroid

        use_oss = self.dataset_folder.startswith('s3://')
        self.text_oss = Text_OSS_IO(disable_client=not use_oss)
        self.npz_oss = Npz_OSS_IO(disable_client=not use_oss)

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')
        self.metadata = self.text_oss.read(metadata_file)
        # metadata is a dict of dict, each element is like '04256520': {'id': '04256520', 'name': 'sofa,couch,lounge'}

        all_categories = [key for key in self.metadata.keys()]
        all_categories = sorted(all_categories)
        # categories is a list of categories like [04256520]

        # Set index
        # we make sure that a category always has the same number label no matter how many categories are actually used
        for c_idx, c in enumerate(all_categories):
            self.metadata[c]['idx'] = c_idx
        # now each element in metadata is like '04256520': {'id': '04256520', 'name': 'sofa,couch,lounge', 'idx': 5}

        if categories is None:
            # categories is a list of categories like ['02691156', '02828884', '02933112']
            categories = all_categories

        if isinstance(split, str):
            self.split_list = [split]
        elif split is None:
            self.split_list = ['train', 'val', 'test']
        else:
            raise Exception('split is not supported:', split)
        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            
            for split in self.split_list:
                split_file = os.path.join(subpath, split + '.lst')
                models_c = self.text_oss.read(split_file).split('\n')
                
                if '' in models_c:
                    models_c.remove('')

                self.models += [{'category': c, 'model': m} for m in models_c]
        # self.models is a list, each element is like {'category': '02958343', 'model': '31055873d40dc262c7477eb29831a699'}

        self.world_size = world_size

        if repeat_dataset > 1:
            one_copy_models = copy.deepcopy(self.models)
            for _ in range(repeat_dataset-1):
                self.models = self.models + copy.deepcopy(one_copy_models)
            print('The dataset has been repeated %d times' % repeat_dataset, flush=True)

        if random_subsample:
            self.models = random.sample(self.models, num_samples)
        whole_dataset_length = len(self.models)
        if world_size > 1:
            if shuffle_before_rank_split:
                random.shuffle(self.models)
            if not whole_dataset_length % world_size == 0:
                print('The number of shapes is %d, which can not be mod by the world size %d' % (whole_dataset_length, world_size))
            self.num_samples_per_rank = int(np.ceil(whole_dataset_length / world_size))
            start = rank * self.num_samples_per_rank
            end = (rank+1) * self.num_samples_per_rank
            if rank == world_size-1:
                rank_idx = np.arange(start, whole_dataset_length)
                missing = end - whole_dataset_length
                if missing > 0 and append_samples_to_last_rank:
                    supp_idx = random.sample(list(range(whole_dataset_length)), missing)
                    supp_idx = np.array(supp_idx)
                    rank_idx = np.concatenate([rank_idx, supp_idx], axis=0)
                    print('%d samples are appended to the last rank' % missing)
            else:
                rank_idx = np.arange(start, end)

            this_rank_models = [self.models[ridx] for ridx in rank_idx]
            self.models = this_rank_models
        else:
            self.num_samples_per_rank = whole_dataset_length

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx'] # the number label of the this category
        category_name = self.metadata[category]['name'].split(',')[0]

        model_path = os.path.join(self.dataset_folder, category, model)
        pointcloud_path = os.path.join(model_path, 'pointcloud.npz')
        data = {}

        # pointcloud_dict = np.load(pointcloud_path)
        start = time.time()
        pointcloud_dict = self.npz_oss.read(pointcloud_path, update_cache=True)
        points = pointcloud_dict['points'].astype(np.float32) # of shape (10 0000, 3), roughly in the range of -0.5 to 0.5
        normals = pointcloud_dict['normals'].astype(np.float32) # of shape (10 0000, 3), the normals are normalized
        # print('points read time %.4f' % (time.time()-start))

        point_idx = random.sample(range(points.shape[0]), self.num_gt_points)
        point_idx = np.array(point_idx)
        points = points[point_idx, :] 
        normals = normals[point_idx, :]
        if self.centered_to_centroid:
            centroid = np.mean(points, axis=0, keepdims=True)
            points = points - centroid

        start = time.time()
        # if self.noise_magnitude > 0:
        #     points = points + self.noise_magnitude * np.random.randn(*points.shape).astype(np.float32)
        #     normals = normals + self.noise_magnitude * np.random.randn(*normals.shape).astype(np.float32)
        #     # in this case, the normals are not normalized, we may need to normalize it after it is gathered into cuda tensors 
        #     # print('noise adding time %.4f' % (time.time()-start))
        points, normals = augment_points_with_normal(points, normals, self.augmentation)
        
        points = points * self.scale * 2 # now it roughly range from -scale to scale

        start = time.time()
        if self.load_psr:
            psr_path = os.path.join(model_path, 'psr.npz')
            psr_dict = self.npz_oss.read(psr_path, update_cache=True)
            psr = psr_dict['psr'].astype(np.float32) # of shape 128,128,128, and it strictly range from -1 to 1
            data['psr'] = psr
            # print('psr read time %.4f' % (time.time()-start))

        data['points'] = points 
        data['normals'] = normals 
        data['label'] = c_idx
        data['category'] = category
        data['category_name'] = category_name
        data['model'] = model

        return data

def augment_points_with_normal(points, normals, augmentation):
    # points and normals are numpy arrays of shape N,3
    if isinstance(augmentation, dict):
        if augmentation.get('mirror_prob',0) > 0:
            if random.random() < augmentation.get('mirror_prob',0):
                axis=2
                center = np.mean(points, axis=0, keepdims=True) # 1,3
                points = points - center
                points[:,axis] = -points[:,axis]
                points = points + center
                normals[:, axis] = -normals[:,axis]
        if augmentation.get('noise_magnitude',0) > 0:
            points = points + augmentation.get('noise_magnitude',0) * np.random.randn(*points.shape).astype(points.dtype)
            normals = normals + augmentation.get('noise_magnitude',0) * np.random.randn(*normals.shape).astype(normals.dtype)
        
        if augmentation.get('translation_magnitude', 0) > 0:
            noise = np.random.normal(scale=augmentation.get('translation_magnitude', 0), size=(1, 3))
            noise = noise.astype(points.dtype)
            points = points + noise

        if augmentation.get('augm_scale', 0) > 1:
            s = random.uniform(1/augmentation.get('augm_scale', 0), augmentation.get('augm_scale', 0))
            points = points * s

    return points, normals
            





if __name__ == '__main__':
    dataset_folder='./shapenet_psr'
    # metadata_file = 'metadata.yaml'
    # dataset_folder='s3://zylyu_datasets/shapenet_psr/'
    # split='train'
    split=None
    categories=None 

    dataset = Shapes3dDataset(dataset_folder, split, categories, rank=0, world_size=1, load_psr=True)
    print(len(dataset))

    # pdb.set_trace()

    # for i in range(len(dataset)):
    #     data = dataset.__getitem__(i)
    #     print('Progress [%d|%d] % .3f' % (i, len(dataset), i/len(dataset)), data['points'].shape, data['normals'].shape, data['psr'].shape, data['label'])
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)
    pdb.set_trace()

    for i, data in enumerate(dataloader):
        print('Progress [%d|%d] % .3f' % (i, len(dataloader), i/len(dataloader)), 
                data['points'].shape, data['normals'].shape, data['psr'].shape, data['label'].shape)
    
    pdb.set_trace()