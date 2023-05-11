import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

from dpsr_utils.utils import mc_from_psr
from dpsr_utils.io_utils import save_mesh, batch_save_pcd, batch_pynt_save_pcd
from data_utils.mirror_partial import mirror_and_concat
from util import AverageMeter
from models.point_upsample_module import point_upsample
from data_utils.points_sampling import sample_keypoints
import pytorch3d
from pytorch3d.ops.utils import masked_gather

import pdb

def shapenet_psr_normalize(x):
    # x is tensor of shape B,N,3
    # we normalize it to shapenet psr dataset scale
    assert x.shape[2] == 3
    minn = x.min(dim=1, keepdim=True)[0] # B,1,3
    maxx = x.max(dim=1, keepdim=True)[0] # B,1,3
    center = (maxx + minn) / 2 # B,1,3
    length = maxx - minn # B,1,3
    max_length = length.max(dim=2, keepdim=True)[0] # B,1,1
    x = (x - center) / max_length * 0.99
    return x

def compute_center_and_max_length(x):
    # x is tensor of shape B,N,3
    # we normalize it to shapenet psr dataset scale
    assert x.shape[2] == 3
    minn = x.min(dim=1, keepdim=True)[0] # B,1,3
    maxx = x.max(dim=1, keepdim=True)[0] # B,1,3
    center = (maxx + minn) / 2 # B,1,3
    length = maxx - minn # B,1,3
    max_length = length.max(dim=2, keepdim=True)[0] # B,1,1
    return center, max_length


def network_output_to_dpsr_grid(X, displacement, dpsr, scale, pointnet_config, 
                    last_dim_as_indicator=False, only_original_points_split=False,
                    explicit_normalize=False):
    '''
        X is of shape B,npoints,F
        displacement is of shape (B, npoints, F*point_upsample_factor)
        dpsr is a DPSR pbject
        scale is the dataset scale
        if last_dim_as_indicator is true, the last feature dim of X is 1 and -1, indicates whether it is an original point
            or it is a mirrored point
        if only_original_points_split, only the original points are splitted, mirrored points are dumped 
    '''
    if last_dim_as_indicator:
        X_to_refine = X[:,:,0:-1]
        if only_original_points_split:
            # we assume the first half points are original points, the rest are mirrored points 
            npoints = X.shape[1]
            npoints = int(npoints/2)
            X_to_refine = X_to_refine[:,0:npoints,:]
            displacement = displacement[:,0:npoints,:]
    else:
        X_to_refine = X
    refined_X = point_upsample(X_to_refine, displacement, pointnet_config['point_upsample_factor'], 
                    include_displacement_center_to_final_output=pointnet_config['include_displacement_center_to_final_output'],
                    output_scale_factor_value=pointnet_config['output_scale_factor'], 
                    first_refine_coarse_points=pointnet_config['first_refine_coarse_points'])
    # refined_X is of shape (B, npoints*point_upsample_factor, 6)

    refined_points = refined_X[:,:,0:3] # (B, npoints*point_upsample_factor, 3)
    refined_normals = refined_X[:,:,3:] # (B, npoints*point_upsample_factor, 3)

    if explicit_normalize:
        # transform to shapenet psr scale
        refined_points = shapenet_psr_normalize(refined_points)
    else:
        refined_points = refined_points / scale / 2 # transform to its original scale
    refined_points = torch.clamp(refined_points / 1.2 + 0.5, min=0, max=0.99) # transformation for dpsr
    
    psr_grid = dpsr(refined_points, refined_normals)

    return psr_grid, refined_points, refined_normals

def evaluate_per_rank(net, dpsr, eval_dataloader, pointnet_config, dpsr_config, trainset_config, dataset, 
        save_dir, iteration, epoch, scale=1, rank=0, world_size=1, use_autoencoder=False, autoencoder=None, noise_magnitude=0):
    assert dataset == 'shapenet_psr_dataset'
    if dataset == 'shapenet_psr_dataset':
        save_file = os.path.join(save_dir, 'shapenet_psr_dpsr_eval_result.pkl')
    print('Eval results will be saved to', save_file)

    total_len = len(eval_dataloader)
    loss_meter = AverageMeter(world_size=world_size)
    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(eval_dataloader):
            X = data['points'].cuda() # of shape (B, npoints, 3), roughly in the range of -scale to scale
            if use_autoencoder:
                if trainset_config['keypoints_source'] == 'farthest_points_sampling':
                    keypoint, _ = sample_keypoints(X, K=trainset_config['num_keypoints'], add_centroid=True)
                else:
                    raise Exception('Only support farthest_points_sampling')
            normals = data['normals'].cuda() # of shape (B, npoints, 3), the normals are normalized
            normals = normals / torch.norm(normals, p=2, dim=2, keepdim=True)
            label = data['label'].cuda()
            psr_gt = data['psr'].cuda()
            if trainset_config.get('include_normals', True):
                X = torch.cat([X, normals], dim=2)
            else:
                X = torch.cat([X, torch.zeros_like(X)], dim=2)
                # in this case, we assume the input point cloud do not have normals, the refinement network need to estimate normals
            condition = None

            if use_autoencoder:
                feature_at_keypoint = autoencoder.encode(X, keypoint, ts=None, label=label, sample_posterior=True)
                X = autoencoder.decode(keypoint, feature_at_keypoint, ts=None, label=label)
                if noise_magnitude > 0:
                    if dpsr_config.get('split_before_refine', False):
                        split_factor = dpsr_config['split_factor']
                        B,N,FF = X.shape
                        noise = noise_magnitude * torch.randn(B,N,split_factor,FF, dtype=X.dtype, device=X.device)
                        X = X.unsqueeze(2) + noise # B,N,split_factor,F
                        X = X.view(B, -1, FF).contiguous()
                    else:
                        X = X + noise_magnitude * torch.randn_like(X)

            mirror_before_upsampling = dpsr_config.get('mirror_before_upsampling', False)
            only_original_points_split = dpsr_config.get('only_original_points_split', False)
            if mirror_before_upsampling:
                X = mirror_and_concat(X, axis=2, num_points=[], attach_label=True, permute=not only_original_points_split)[0]

            displacement = net(X, condition, ts=None, label=label) # (B, npoints, 6*point_upsample_factor)
            psr_grid, _, _ = network_output_to_dpsr_grid(X, displacement, dpsr, scale, pointnet_config,
                    last_dim_as_indicator=mirror_before_upsampling, only_original_points_split=only_original_points_split)
            
            loss = F.mse_loss(psr_grid, psr_gt)
            loss_meter.update(loss.cpu().item(), n=psr_gt.shape[0])
        print('This rank loss is %.6f' % loss_meter.avg)
        if world_size > 1:
            torch.distributed.barrier()
        total_sums, total_count = loss_meter.tensor_reduce()
        if rank == 0:
            reduced_loss = (total_sums / total_count).cpu().item()
            print('gathered all rank loss is %.6g' % reduced_loss)
            current_results = {'iter':iteration, 'dpsr_grid_L2_loss':reduced_loss, 'epoch':epoch}
            merge_current_with_previous_eval_results(current_results, save_file)
    net.train()
    

def merge_current_with_previous_eval_results(current_results, save_file):
    # current_results is a dict that contains all eval result at this itertion
    # we merge it with previous saved evaluation result file, so that we can draw a training process curve
    if os.path.isfile(save_file):
        handle = open(save_file, 'rb')
        data = pickle.load(handle)
        handle.close()
        for key in data.keys():
            data[key].append(current_results[key])
    else:
        data = {}
        for key in current_results.keys():
            data[key] = [current_results[key]]
    handle = open(save_file, 'wb')
    pickle.dump(data, handle)
    handle.close()
    print('Eval results have been saved to', save_file)

    save_dir = os.path.split(save_file)[0]
    save_dir = os.path.join(save_dir, 'figures')
    plot_result(data, 'iter', save_dir, plot_values=None, print_lowest_value=True)
    

def visualize_per_rank(net, dpsr, vis_dataloader, pointnet_config, dpsr_config, trainset_config, dataset, 
        save_dir, iteration, epoch, scale=1, rank=0, world_size=1, use_autoencoder=False, autoencoder=None, noise_magnitude=0,
        sample_points_from_mesh=False, explicit_normalize=False, label_number=None, return_original_scale=False):
    assert dataset == 'shapenet_psr_dataset'
    # if dataset == 'shapenet_psr_dataset':
    #     save_file = os.path.join(save_dir, 'shapenet_psr_dpsr_eval_result.pkl')

    total_len = len(vis_dataloader)
    # loss_meter = AverageMeter(world_size=world_size)

    vis_save_dir = 'visualization_results_at_iteration_%s_epoch_%s' % (str(iteration).zfill(8), str(epoch).zfill(4))
    print('visualization results will be saved to the folder',  os.path.join(save_dir, vis_save_dir))
    # pdb.set_trace()
    noisy_pcd_save_dir = os.path.join(save_dir, vis_save_dir, 'noisy_pcd')
    refined_pcd_save_dir = os.path.join(save_dir, vis_save_dir, 'refined_pcd')
    reconstructed_mesh_save_dir = os.path.join(save_dir, vis_save_dir, 'reconstructed_mesh')
    os.makedirs(noisy_pcd_save_dir, exist_ok=True)
    os.makedirs(refined_pcd_save_dir, exist_ok=True)
    os.makedirs(reconstructed_mesh_save_dir, exist_ok=True)
    if sample_points_from_mesh:
        point_from_mesh_save_dir = os.path.join(save_dir, vis_save_dir, 'points_sampled_from_mesh')
        os.makedirs(point_from_mesh_save_dir, exist_ok=True)
        uniform_point_from_mesh_save_dir = os.path.join(save_dir, vis_save_dir, 'uniform_points_sampled_from_mesh')
        os.makedirs(uniform_point_from_mesh_save_dir, exist_ok=True)
        result = {'points_sampled_from_mesh': [], 'normals_sampled_from_mesh': [], 
                    'uniform_points_sampled_from_mesh': [], 'uniform_normals_sampled_from_mesh': [], 'label':[], }


    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(vis_dataloader):
            X = data['points'].cuda() # of shape (B, npoints, 3), roughly in the range of -scale to scale
            original_center, original_max_length = compute_center_and_max_length(X)
            if use_autoencoder:
                if trainset_config['keypoints_source'] == 'farthest_points_sampling':
                    keypoint, _ = sample_keypoints(X, K=trainset_config['num_keypoints'], add_centroid=True)
                else:
                    raise Exception('Only support farthest_points_sampling')
            
            if 'label' in data.keys():
                label = data['label'].cuda()
            else:
                label = (label_number * torch.ones(X.shape[0]).cuda()).long()
            # psr_gt = data['psr'].cuda()
            if trainset_config.get('include_normals', True):
                normals = data['normals'].cuda() # of shape (B, npoints, 3), the normals are normalized
                normals = normals / torch.norm(normals, p=2, dim=2, keepdim=True)
                X = torch.cat([X, normals], dim=2)
            else:
                X = torch.cat([X, torch.zeros_like(X)], dim=2)
                # in this case, we assume the input point cloud do not have normals, the refinement network need to estimate normals
            condition = None
            category_name = data.get('category_name', None)

            if use_autoencoder:
                feature_at_keypoint = autoencoder.encode(X, keypoint, ts=None, label=label, sample_posterior=True)
                X = autoencoder.decode(keypoint, feature_at_keypoint, ts=None, label=label)
                if noise_magnitude > 0:
                    if dpsr_config.get('split_before_refine', False):
                        split_factor = dpsr_config['split_factor']
                        B,N,FF = X.shape
                        noise = noise_magnitude * torch.randn(B,N,split_factor,FF, dtype=X.dtype, device=X.device)
                        X = X.unsqueeze(2) + noise # B,N,split_factor,F
                        X = X.view(B, -1, FF).contiguous()
                    else:
                        X = X + noise_magnitude * torch.randn_like(X)

            mirror_before_upsampling = dpsr_config.get('mirror_before_upsampling', False)
            only_original_points_split = dpsr_config.get('only_original_points_split', False)
            if mirror_before_upsampling:
                X = mirror_and_concat(X, axis=2, num_points=[], attach_label=True, permute=not only_original_points_split)[0]

            displacement = net(X, condition, ts=None, label=label) # (B, npoints, 6*point_upsample_factor)
            psr_grid, refined_points, refined_normals = network_output_to_dpsr_grid(X, displacement, dpsr, scale, pointnet_config, 
                last_dim_as_indicator=mirror_before_upsampling, only_original_points_split=only_original_points_split,
                explicit_normalize=explicit_normalize)

            start_idx = vis_dataloader.dataset.num_samples_per_rank * rank
            start_idx = start_idx + vis_dataloader.batch_size * idx

            indicator = X[:,:,-1] if mirror_before_upsampling else None
            batch_pynt_save_pcd(noisy_pcd_save_dir, 'noisy_pcd', X[:,:,0:3], batch_info=category_name, 
                                normals=X[:,:,3:6], indicator=indicator, start_idx=start_idx)
            batch_pynt_save_pcd(refined_pcd_save_dir, 'refined_pcd', refined_points, batch_info=category_name, 
                                normals=refined_normals, start_idx=start_idx)
            points_from_mesh, normals_from_mesh, uniform_points_from_mesh, uniform_normals_from_mesh = batch_mc_from_psr(
                                            psr_grid, reconstructed_mesh_save_dir, 'reconstructed_mesh', 
                                            batch_info=category_name, start_idx=start_idx,
                                            sample_points_from_mesh=sample_points_from_mesh,
                                            return_original_scale=return_original_scale, 
                                            original_center=original_center, 
                                            original_max_length=original_max_length)
            if sample_points_from_mesh:
                batch_pynt_save_pcd(point_from_mesh_save_dir, 'pcd_from_mesh', points_from_mesh, batch_info=category_name, 
                                normals=normals_from_mesh, start_idx=start_idx)
                batch_pynt_save_pcd(uniform_point_from_mesh_save_dir, 'pcd_from_mesh', uniform_points_from_mesh, batch_info=category_name, 
                                normals=uniform_normals_from_mesh, start_idx=start_idx)
                
                batch_result = (points_from_mesh, normals_from_mesh, uniform_points_from_mesh, uniform_normals_from_mesh, 
                                label.detach().cpu().numpy())
                for i, key in enumerate(result):
                    result[key].append(batch_result[i])
    if sample_points_from_mesh:
        # note that in this case you need to assure world_size==1
        # because we do not explicitly gather npz files from different ranks
        for key in result.keys():
            result[key] = np.concatenate(result[key], axis=0)
        save_name = os.path.join(save_dir, vis_save_dir, 'points_sampled_from_mesh.npz')
        np.savez(save_name, points=result['points_sampled_from_mesh'], 
                    normals=result['normals_sampled_from_mesh'], label=result['label'])
        save_name = os.path.join(save_dir, vis_save_dir, 'uniform_points_sampled_from_mesh.npz')
        np.savez(save_name, points=result['uniform_points_sampled_from_mesh'], 
                    normals=result['uniform_normals_sampled_from_mesh'], label=result['label'])
    net.train()

def batch_mc_from_psr(psr_grid, save_dir, save_prefix, batch_info=None, start_idx = 0, sample_points_from_mesh=False, 
            return_original_scale=False, original_center=None, original_max_length=None):
    # run marching cube for psr_grid and save generated meshes
    # psr_grid is of shape (B, res, res, res)
    B = psr_grid.shape[0]
    total_points = []
    total_normals = []
    total_points_uniform = []
    total_normals_uniform = []
    for i in range(B):
        psr_grid_i = psr_grid[i:i+1]
        v, f, n = mc_from_psr(psr_grid_i, zero_level=0)
        # v, f, n are list of length 1, contains numpy arrays
        v, f, n = v[0], f[0], n[0] #
        if return_original_scale:
            verts = torch.from_numpy(np.ascontiguousarray(v)).cuda().unsqueeze(0)
            center, max_length = compute_center_and_max_length(verts)
            verts = (verts - center) / max_length * original_max_length[i:i+1] + original_center[i:i+1]
            v = verts[0].cpu().numpy()
        if sample_points_from_mesh:
            verts = torch.from_numpy(np.ascontiguousarray(v)).cuda().unsqueeze(0)
            faces = torch.from_numpy(np.ascontiguousarray(f)).cuda().unsqueeze(0)
            normals = torch.from_numpy(np.ascontiguousarray(n)).cuda().unsqueeze(0)
            mesh = pytorch3d.structures.Meshes(verts=verts, faces=faces, verts_normals=normals)
            # result = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=2048, return_normals=True)
            points, normals = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=2048, return_normals=True)
            # points and normals are of shape 1,2048,3
            total_points.append(points.detach().cpu().numpy())
            total_normals.append(normals.detach().cpu().numpy())

            points_dense, normals_dense = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=20480, return_normals=True)
            points_uniform, selected_idx = pytorch3d.ops.sample_farthest_points(points_dense, 
                                                    K=2048, random_start_point=True)
            normals_uniform = masked_gather(normals_dense, selected_idx)

            total_points_uniform.append(points_uniform.detach().cpu().numpy())
            total_normals_uniform.append(normals_uniform.detach().cpu().numpy())
        if batch_info is None:
            save_name = save_prefix + '_' + str(start_idx+i).zfill(5) + '.ply'
        else:
            save_name = batch_info[i] + '_' + str(start_idx+i).zfill(5) + '.ply'
        save_name = os.path.join(save_dir, save_name)
        save_mesh(save_name, v, f, normals=n)

    if sample_points_from_mesh:
        total_points = np.concatenate(total_points, axis=0)
        total_normals = np.concatenate(total_normals, axis=0)
        total_points_uniform = np.concatenate(total_points_uniform, axis=0)
        total_normals_uniform = np.concatenate(total_normals_uniform, axis=0)
    return total_points, total_normals, total_points_uniform, total_normals_uniform


def find_and_print_lowest_value(x,y, x_key, y_key):
    idx = np.argmin(y)
    x_min = x[idx]
    y_min = y[idx]
    print('The lowest value of %s is %.8f at %s %.2f' % (y_key, y_min, x_key, x_min), flush=True)

def plot_result(result, plot_key, save_dir, plot_values=None, print_lowest_value=False):
    # result is a dictionary of lists
    # result[plot_key] is the horizontal axis
    # result[key] is vertical axis
    # we plot all other keys except plot_key against plot_key in result if plot_values is None
    # plot_values could aslo be a list of keys
    # we only plot those keys specified in plot_values against plot_key
    # print('\n Comparing current ckpt with previous saved ckpts', flush=True)
    os.makedirs(save_dir, exist_ok=True)
    x = np.array(result[plot_key])
    order = np.argsort(x)
    x = x[order]
    if len(result[plot_key]) > 1:
        for key in result.keys():
            plot = not key == plot_key
            if not plot_values is None:
                plot = plot and key in plot_values
            if plot:
                plt.figure()
                if isinstance(result[key], dict):
                    for sub_key in result[key].keys():
                        y = np.array(result[key][sub_key])
                        y = y[order]
                        plt.plot(x, y, marker = '.', label=sub_key)
                        if print_lowest_value:
                            find_and_print_lowest_value(x, y, plot_key, key+'-'+sub_key)
                    plt.xlabel(plot_key)
                    plt.legend()
                else:
                    y = np.array(result[key])
                    y = y[order]
                    plt.plot(x, y, marker = '.')
                    plt.xlabel(plot_key)
                    plt.ylabel(key)
                    if print_lowest_value:
                        find_and_print_lowest_value(x, y, plot_key, key)
                save_file = os.path.join(save_dir, key+'.png')
                plt.savefig(save_file)
                plt.close()
                print('have save the figure for %s to the file %s' % (key, save_file), flush=True)
    else:
        print('Do not plot because there is only 1 value in plot key', flush=True)
    return


