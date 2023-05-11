import torch
from torchvision import transforms
import torch.utils.data as data

import numpy as np

# from mvp_dataloader.mvp_dataset import ShapeNetH5
from shapenet_psr_dataloader.shapenet_psr_dataset import Shapes3dDataset

def get_dataloader(args, phase='train', rank=0, world_size=1, append_samples_to_last_rank=True, 
            shuffle_before_rank_split=True, random_subsample=False, num_samples=1000):

    if args['dataset'] == 'shapenet_psr_dataset':
        assert phase in ['train', 'test', 'val']
        if phase == 'train':
            batch_size = int(args['batch_size'] / world_size)
            shuffle = True
        else:
            batch_size = int(args['eval_batch_size'] / world_size)
            shuffle = False
        # if 'augmentation' in args.keys():
        #     noise_magnitude = args['augmentation']['noise_magnitude']
        # else:
        #     noise_magnitude = 0
        repeat_dataset = args.get('repeat_dataset', 1)
        centered_to_centroid = args.get('centered_to_centroid', False)

        dataset = Shapes3dDataset(args['data_dir'], split=phase, categories=args['categories'], scale=args['scale'], 
                        num_gt_points=args['npoints'], rank=rank, world_size=world_size,
                        append_samples_to_last_rank=append_samples_to_last_rank, 
                        shuffle_before_rank_split=shuffle_before_rank_split,
                        load_psr=args.get('load_psr', False),
                        augmentation=args.get('augmentation', False),
                        random_subsample=random_subsample, num_samples=num_samples,
                        repeat_dataset=repeat_dataset,
                        centered_to_centroid=centered_to_centroid)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                                    num_workers=args['num_workers'])
    else:
        raise Exception(args['dataset'], 'dataset is not supported')

    return trainloader


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, length, num_labels=13, rank=0, world_size=1):
        if world_size == 1:
            self.length = length
        else:
            self.length = int(np.ceil(length / world_size))
            if rank == world_size-1:
                self.length = length - (world_size-1) * self.length
        self.num_labels = num_labels

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return self.length

    def __getitem__(self, idx):
        label = np.random.randint(self.num_labels)
        result = {}
        result['label'] = label

        return result



