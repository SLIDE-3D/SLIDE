# import torch
from util import sampling
import numpy as np
import os
# from shapenet_psr_dataloader.dummy_shapenet_psr_dataset import DummyShapes3dDataset
from dataset import get_dataloader
from visualization_tools.visualize_pcd import visualize_pcd
from data_utils.points_sampling import sample_keypoints
from shapenet_psr_dataloader.npz_dataset import GeneralNpzDataset
import torch
import copy
import time
import pdb

def evaluate_per_rank(net, trainset_config, diffusion_hyperparams, save_dir, task, point_feature_dim=3, diffusion_model=None, 
                        rank=0, world_size=1, ckpt_info='', keypoint_dim=3, test_external_keypoint=False, external_keypoint_file=None, 
                        split_points_and_normals=False, include_idx_to_save_name=True, save_keypoint_feature=False,
                        local_resampling=False, complete_x0=None, keypoint_mask=None):
    # external_keypoint_file and test_external_keypoint is only used in the case of keypoint_conditional_generation
    # in this case we want to see the point cloud generation results conditioned on some external keypoints provied in external_keypoint_file

    dataset = trainset_config['dataset']
    assert task in ['generation', 'keypoint_generation', 'keypoint_conditional_generation', 'latent_generation', 'latent_keypoint_conditional_generation']

    if task in ['generation', 'latent_generation', 'keypoint_conditional_generation', 'latent_keypoint_conditional_generation'] :
        num_points = trainset_config['npoints']
    elif task == 'keypoint_generation':
        num_points = trainset_config['num_keypoints']
    # num_points is the number of points in the final generated point cloud

    # num_samples = trainset_config['num_samples_tested'] / world_size
    # batch_size = trainset_config['eval_batch_size'] / world_size
    # dataset_folder = trainset_config['data_dir']
    # categories = trainset_config['categories']
    os.makedirs(save_dir, exist_ok=True)
    
    assert dataset == 'shapenet_psr_dataset'
    if dataset == 'shapenet_psr_dataset':
        if world_size == 1:
            save_file = os.path.join(save_dir, 'shapenet_psr_generated_data_%d_pts%s.npz' % (num_points, ckpt_info))
        else:
            save_file = os.path.join(save_dir, 'shapenet_psr_generated_data_%d_pts_rank_%d%s.npz' % (num_points, rank, ckpt_info))
        # num_labels = 13
    print('Generated samples will be saved to', save_file)

    # set up the dataloader
    
    if test_external_keypoint:
        test_dataset = GeneralNpzDataset(external_keypoint_file, scale=1, noise_magnitude=0, rank=rank, world_size=world_size, 
                                        data_key='points', data_key_split_names=None, data_key_split_dims=None)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(trainset_config['eval_batch_size']/world_size), 
                                        shuffle=False, num_workers=8)
    else:
        trainset_config_copy = copy.deepcopy(trainset_config)
        trainset_config_copy['batch_size'] = trainset_config_copy['eval_batch_size']
        # eval_batch_size and num_samples_tested will be automatically divided by world_size in the dataloader
        testloader = get_dataloader(trainset_config_copy, phase='train', rank=rank, world_size=world_size, 
                        append_samples_to_last_rank=False, shuffle_before_rank_split=False,
                        random_subsample=True, num_samples=trainset_config_copy['num_samples_tested'])

    total_generated_data = []
    total_generated_keypoint = [] # for keypoint conditional generation and latent diffusion generation
    total_generated_keypoint_feature = []
    total_generated_label = []
    total_generated_category = []
    total_generated_category_name = []
    total_gt_points = [] # for keypoint conditional generation
    total_timing = []
    total_len = len(testloader)
    net.eval()
    for idx, data in enumerate(testloader):
        label = data['label'].cuda()
        batch = label.shape[0]
        condition = None
        keypoint = None
        if task in ['keypoint_conditional_generation', 'latent_keypoint_conditional_generation']:
            if test_external_keypoint:
                # in this case, we test external keypoint
                keypoint = data['points'].cuda()
                keypoint_noise_magnitude = trainset_config.get('keypoint_noise_magnitude', 0)
                if keypoint_noise_magnitude > 0:
                    keypoint = keypoint + keypoint_noise_magnitude * torch.randn_like(keypoint)
                condition = keypoint
                condition = keypoint
                total_generated_keypoint.append(keypoint.detach().cpu().numpy())
            else:
                # in this case, gt is known
                gt_points = data['points'].cuda()
                keypoint, _ = sample_keypoints(gt_points, K=trainset_config['num_keypoints'], 
                                add_centroid=trainset_config.get('add_centroid_to_keypoints', True),
                                random_subsample=trainset_config.get('random_subsample', False))
                keypoint_noise_magnitude = trainset_config.get('keypoint_noise_magnitude', 0)
                if keypoint_noise_magnitude > 0:
                    keypoint = keypoint + keypoint_noise_magnitude * torch.randn_like(keypoint)
                condition = keypoint
                gt_points = torch.cat([data['points'], data['normals']], dim=2)
                total_gt_points.append(gt_points.detach().cpu().numpy())
                total_generated_keypoint.append(keypoint.detach().cpu().numpy())

        print('progress [%d/%d] %.4f, %d samples' % (idx, total_len, idx/total_len, batch), flush=True)
        
        start_time = time.time()
        if diffusion_model is None:
            generated_data = sampling(net, (batch,num_points,3+point_feature_dim), 
                                    diffusion_hyperparams, 
                                    print_every_n_steps=200, label=label, 
                                    condition=condition,
                                    verbose=False,
                                    use_a_precomputed_XT=False, step=None, XT=None)
        else:
            if task in ['latent_generation', 'latent_keypoint_conditional_generation']:
                # pdb.set_trace()
                generated_data, generated_keypoint, keypoint_feature = diffusion_model.denoise_and_reconstruct(
                                        batch, net, keypoint_dim, 
                                        (trainset_config['num_keypoints'],3+point_feature_dim), label=label, 
                                        keypoint=None if keypoint is None else keypoint.float(), 
                                        return_keypoint_feature=True,
                                        local_resampling=local_resampling, complete_x0=complete_x0, keypoint_mask=keypoint_mask)
                if save_keypoint_feature:
                    total_generated_keypoint_feature.append(keypoint_feature.detach().cpu().numpy())
                if task == 'latent_generation':
                    # if keypoint_conditional, keypoints are given, we do not need to resave the keypoints
                    total_generated_keypoint.append(generated_keypoint.detach().cpu().numpy())
            else:
                generated_data = diffusion_model.denoise(batch, net, (num_points,3+point_feature_dim), label=label)
        total_timing.extend( [(time.time()-start_time)/batch] * batch )
        total_generated_data.append(generated_data.detach().cpu().numpy())
        total_generated_label.append(label.detach().cpu().numpy())
        total_generated_category = total_generated_category + data['category']
        total_generated_category_name = total_generated_category_name + data['category_name']
    net.train()
    total_generated_data = np.concatenate(total_generated_data, axis=0)
    total_generated_label = np.concatenate(total_generated_label, axis=0)
    total_timing = np.array(total_timing)
    result_dict = {'points': total_generated_data, 'label': total_generated_label, 'category': total_generated_category, 
                'category_name': total_generated_category_name, 'timing':total_timing}
    if len(total_generated_keypoint) > 0:
        total_generated_keypoint = np.concatenate(total_generated_keypoint, axis=0)
        result_dict['keypoint'] = total_generated_keypoint
    if len(total_gt_points) > 0:
        total_gt_points = np.concatenate(total_gt_points, axis=0)
        result_dict['gt_points'] = total_gt_points
    if save_keypoint_feature:
        total_generated_keypoint_feature = np.concatenate(total_generated_keypoint_feature, axis=0)
        result_dict['keypoint_feature'] = total_generated_keypoint_feature
    if split_points_and_normals:
        if result_dict['points'].shape[2] == 6:
            result_dict['normals'] = result_dict['points'][:,:,3:]
            result_dict['points'] = result_dict['points'][:,:,0:3]
    np.savez(save_file, **result_dict)
    print('Generated samples have been saved to', save_file)
    print('The average generation time of a single sample is', total_timing.sum() / result_dict['points'].shape[0])
    if world_size == 1:
        visualize_pcd(save_file, include_idx_to_save_name=include_idx_to_save_name)

def gather_generated_results(dataset, save_dir, world_size, num_points=2048, ckpt_info=''):
    assert dataset == 'shapenet_psr_dataset'
    result_dict = {}
    gathered_files = []

    if dataset == 'shapenet_psr_dataset':
        rank_file_root = os.path.join(save_dir, 'shapenet_psr_generated_data_%d_pts_rank_' % (num_points))
        save_file = os.path.join(save_dir, 'shapenet_psr_generated_data_%d_pts%s.npz' % (num_points, ckpt_info))
        # print('Gathered results will be saved to', save_file)

    for rank in range(world_size):
        rank_file = rank_file_root + '%d%s.npz' % (rank, ckpt_info)
            
        data = np.load(rank_file)
        for file_name in data._files:
            file_name = os.path.splitext(file_name)[0]
            if file_name in result_dict.keys():
                result_dict[file_name].append(data[file_name])
            else:
                result_dict[file_name] = [data[file_name]]

        gathered_files.append(rank_file)

    for key in result_dict.keys():
        result_dict[key] = np.concatenate(result_dict[key], axis=0)
    np.savez(save_file, **result_dict)
    print('Gathered results have been saved to', save_file)
    visualize_pcd(save_file)

    for f in gathered_files:
        os.remove(f)


