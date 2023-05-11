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
    uniform_fps = False 
    split='val'
    random_subsample=False
    num_samples=256
    save_dir = '/mnt/cache/lvzhaoyang/mesh/Point_Diffusion_for_Mesh/pointnet2/exps/shapenet_psr_validation_set'
    if uniform_fps:
        save_dir = os.path.join(save_dir, 'uniform_pcds')
    os.makedirs(save_dir, exist_ok=True)
    # categories=['02691156']

    text_oss = Text_OSS_IO(disable_client=False)
    metadata_file = os.path.join(dataset_folder, 'metadata.yaml')
    metadata = text_oss.read(metadata_file)
    # pdb.set_trace()
    for key in metadata.keys():
        if key == '03636649':
            category = metadata[key]['id']
            category_name = metadata[key]['name'].split(',')[0]
            dataset = Shapes3dDataset(dataset_folder, split, [category], rank=0, world_size=1, load_psr=False, 
                        num_gt_points=100000 if uniform_fps else num_gt_points, 
                        random_subsample=random_subsample, num_samples=num_samples)
            print(category_name, len(dataset))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
            result_dict = {}
            
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    if uniform_fps:
                        assert 'points' in data.keys() and 'normals' in data.keys()
                        data['points'] = data['points'].cuda()
                        data['normals'] = data['normals'].cuda()
                        selected_points, selected_idx = pytorch3d.ops.sample_farthest_points(data['points'], 
                                                        K=num_gt_points, random_start_point=True)
                        data['normals'] = masked_gather(data['normals'], selected_idx).detach().cpu()
                        data['points'] = selected_points.detach().cpu()
                    for key in data.keys():
                        if key in result_dict.keys():
                            result_dict[key].append(data[key])
                        else:
                            result_dict[key] = [data[key]]
            # pdb.set_trace()
            for key in result_dict:
                # if isinstance(result_dict[key][0], np.ndarray):
                #     result_dict[key] = np.concatenate(result_dict[key])
                if isinstance(result_dict[key][0], torch.Tensor):
                    result_dict[key] = torch.cat(result_dict[key], dim=0).detach().cpu().numpy()
                else:
                    result_dict[key] = sum(result_dict[key], [])
            for key in result_dict.keys():
                if isinstance(result_dict[key], np.ndarray):
                    print(key, result_dict[key].shape)
                else:
                    print(key, len(result_dict[key]))
            save_name = category_name + '_' + category + '_%d_samples.npz' % len(dataset)
            save_name = os.path.join(save_dir, save_name) 
            np.savez(save_name, **result_dict)
            print('validation set for %s has been saved to %s' % (category_name, save_name))
            # pdb.set_trace()
    
    