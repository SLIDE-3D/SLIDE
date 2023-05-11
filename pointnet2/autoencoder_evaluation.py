import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

from util import AverageMeter
from data_utils.points_sampling import sample_keypoints
from visualization_tools.visualize_hierarchical_pcd import visualize_hierarchical_pcd
from dpsr_evaluation import merge_current_with_previous_eval_results

def evaluate_per_rank(net, eval_dataloader, dataset, save_dir, iteration, epoch, trainset_config, rank=0, world_size=1, 
                    save_reconstructed_pcd=False, keypoint_source='farthest_points_sampling', save_keypoint_feature=False):
    if save_keypoint_feature:
        assert save_reconstructed_pcd
    assert dataset == 'shapenet_psr_dataset'
    if dataset == 'shapenet_psr_dataset':
        save_file = os.path.join(save_dir, 'shapenet_psr_autoencoder_visualization_result_iteration_%s_epoch_%s_rank_%d.pkl' % 
                                (str(iteration).zfill(8), str(epoch).zfill(4), rank))
    print('Eval results will be saved to', save_file)

    # total_len = len(eval_dataloader)
    # loss_meter = AverageMeter(world_size=world_size)
    net.eval()

    total_xyz = None
    total_gt_points = []
    total_generated_label = []
    total_generated_category = []
    total_generated_category_name = []
    total_generated_model = []
    total_generated_keypoint_feature = []
    total_generated_keypoint = []
    with torch.no_grad():
        for idx, data in enumerate(eval_dataloader):
            points = data['points'].cuda() # of shape (B, npoints, 3), roughly in the range of -scale to scale
            normals = data['normals'].cuda() # of shape (B, npoints, 3), the normals are normalized
            normals = normals / torch.norm(normals, p=2, dim=2, keepdim=True)
            label = data['label'].cuda()
            # keypoints, _ = sample_keypoints(points, K=num_keypoints, add_centroid=True)
            if keypoint_source == 'farthest_points_sampling':
                keypoints, _ = sample_keypoints(points, K=trainset_config['num_keypoints'], 
                                add_centroid=trainset_config.get('add_centroid_to_keypoints', True),
                                random_subsample=trainset_config.get('random_subsample', False))
            else:
                keypoints = data['keypoint'].float().cuda()
            keypoint_noise_magnitude = trainset_config.get('keypoint_noise_magnitude', 0)
            if keypoint_noise_magnitude > 0:
                keypoints = keypoints + keypoint_noise_magnitude * torch.randn_like(keypoints)
            X = torch.cat([points, normals], dim=2)
            
            total_generated_keypoint.append(keypoints.detach().cpu().numpy())
            l_xyz, loss_list, keypoint_feature = net(X, keypoints, ts=None, label=label, 
                                                    loss_type='cd_p', return_keypoint_feature=True)
            if save_keypoint_feature:
                total_generated_keypoint_feature.append(keypoint_feature.detach().cpu().numpy())
            if total_xyz is None:
                total_xyz = [[xyz.detach().cpu().numpy()] for xyz in l_xyz]
            else:
                for i in range(len(l_xyz)):
                    total_xyz[i].append(l_xyz[i].detach().cpu().numpy())
            total_gt_points.append(points.detach().cpu().numpy())
            total_generated_label.append(label.detach().cpu().numpy())
            total_generated_category = total_generated_category + data['category']
            total_generated_category_name = total_generated_category_name + data['category_name']
            total_generated_model = total_generated_model + data['model']
    
    total_generated_label = np.concatenate(total_generated_label, axis=0)
    total_gt_points = np.concatenate(total_gt_points, axis=0)
    total_generated_keypoint = np.concatenate(total_generated_keypoint, axis=0)
    if save_keypoint_feature:
        total_generated_keypoint_feature = np.concatenate(total_generated_keypoint_feature, axis=0)
    total_xyz = [ np.concatenate(xyz, axis=0) for xyz in total_xyz ]

    handle = open(save_file, 'wb')
    pickle.dump({'hierarchical_pointcloud':total_xyz, 'label': total_generated_label, 'category': total_generated_category, 
                'category_name': total_generated_category_name, 'gt_points':total_gt_points, 'model':total_generated_model}, handle)
    handle.close()
    if save_reconstructed_pcd:
        assert world_size==1
        result = {'points':total_xyz[-1][:,:,0:3], 'label':total_generated_label, 'category':total_generated_category, 
                'category_name':total_generated_category_name, 'model':total_generated_model, 
                'keypoint':total_generated_keypoint}
        if total_xyz[-1].shape[2] == 6:
            result['normals'] = total_xyz[-1][:,:,3:6]
        if save_keypoint_feature:
            result['keypoint_feature'] = total_generated_keypoint_feature
        pcd_save_file = os.path.join(save_dir, 'reconstructed_pcd.npz')
        np.savez(pcd_save_file, **result)
        print('reconstructed_pcd has been saved to', pcd_save_file)
        
    if world_size==1:
        visualize_hierarchical_pcd(save_file)
    else:
        torch.distributed.barrier()
        if rank==0:
            rank_file_root = 'shapenet_psr_autoencoder_visualization_result_iteration_%s_epoch_%s' % (
                                                        str(iteration).zfill(8), str(epoch).zfill(4))
            total_save_file = 'shapenet_psr_autoencoder_visualization_result_iteration_%s_epoch_%s.pkl' % (
                                                        str(iteration).zfill(8), str(epoch).zfill(4))
            gather_generated_results(dataset, save_dir, rank_file_root, total_save_file, world_size)
    net.train()

def gather_generated_results(dataset, save_dir, rank_file_root, save_file, world_size):
    assert dataset == 'shapenet_psr_dataset'
    result_dict = {}
    gathered_files = []

    save_file = os.path.join(save_dir, save_file)

    for rank in range(world_size):
        rank_file = rank_file_root + '_rank_%d.pkl' % (rank)
        rank_file = os.path.join(save_dir, rank_file)
        handle = open(rank_file, 'rb')
        data = pickle.load(handle)
        handle.close()
        # we assume the values in data are numpy arrays, or list of numpy arrays, or list of strings 
        for key in data.keys():
            if key in result_dict.keys():
                if isinstance(data[key], np.ndarray):
                    result_dict[key] = np.concatenate([result_dict[key], data[key]], axis=0)
                elif isinstance(data[key], list) and isinstance(data[key][0], str):
                    result_dict[key] = result_dict[key] + data[key]
                elif isinstance(data[key], list) and isinstance(data[key][0], np.ndarray):
                    for i in range(len(data[key])):
                        result_dict[key][i] = np.concatenate([result_dict[key][i], data[key][i]], axis=0)
                else:
                    print(key, type(data[key]), data[key])
                    raise Exception('not supported type')
            else:
                result_dict[key] = data[key]
            
        gathered_files.append(rank_file)

    handle = open(save_file, 'wb')
    pickle.dump(result_dict, handle)
    handle.close()
    print('Gathered results have been saved to', save_file)
    visualize_hierarchical_pcd(save_file)

    for f in gathered_files:
        os.remove(f)

def quantitative_evaluate_per_rank(net, eval_dataloader, dataset, save_dir, iteration, epoch, trainset_config, rank=0, world_size=1):
    assert dataset == 'shapenet_psr_dataset'
    if dataset == 'shapenet_psr_dataset':
        save_file = os.path.join(save_dir, 'shapenet_psr_autoencoder_quantitative_eval_result.pkl')
    print('Eval results will be saved to', save_file)
    os.makedirs(save_dir, exist_ok=True)

    # total_len = len(eval_dataloader)
    # loss_meter = AverageMeter(world_size=world_size)
    # assert world_size == 1
    net.eval()
    avg_meter_dict = None
    
    with torch.no_grad():
        for idx, data in enumerate(eval_dataloader):
            points = data['points'].cuda() # of shape (B, npoints, 3), roughly in the range of -scale to scale
            normals = data['normals'].cuda() # of shape (B, npoints, 3), the normals are normalized
            normals = normals / torch.norm(normals, p=2, dim=2, keepdim=True)
            label = data['label'].cuda()
            # keypoints, _ = sample_keypoints(points, K=num_keypoints, add_centroid=True)
            keypoints, _ = sample_keypoints(points, K=trainset_config['num_keypoints'], 
                                add_centroid=trainset_config.get('add_centroid_to_keypoints', True),
                                random_subsample=trainset_config.get('random_subsample', False))
            keypoint_noise_magnitude = trainset_config.get('keypoint_noise_magnitude', 0)
            if keypoint_noise_magnitude > 0:
                keypoints = keypoints + keypoint_noise_magnitude * torch.randn_like(keypoints)
            X = torch.cat([points, normals], dim=2)

            l_xyz, loss_list = net(X, keypoints, ts=None, label=label, loss_type='cd_p')
            if avg_meter_dict is None:
                avg_meter_dict = {}
                for key in loss_list[0].keys():
                    avg_meter_dict[key] = AverageMeter(world_size=world_size)
            for key in loss_list[0].keys():
                # we only record the last level loss
                avg_meter_dict[key].update(loss_list[-1][key].mean().cpu().item(), n=points.shape[0])

    current_results = {'iter':iteration, 'epoch':epoch}
    if world_size > 1:
        torch.distributed.barrier()
    for key in avg_meter_dict.keys():
        total_sums, total_count = avg_meter_dict[key].tensor_reduce()
        current_results[key] = (total_sums / total_count).cpu().item()
    if rank == 0:
        merge_current_with_previous_eval_results(current_results, save_file)
            
    net.train()


