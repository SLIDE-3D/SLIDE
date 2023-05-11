import torch
import copy
from pointnet2_ops import pointnet2_utils
import numpy as np
import os
import pdb

def mirror(partial, axis=1):
    # partial is of shape B,N,F, we assume it first 3 dimensions are 3d xyz coordinates
    # its next 3 dimensions are 3d normals, it could also contain other features
    xyz = partial[:,:,0:3] # B,N,3
    center = torch.mean(xyz, dim=1, keepdim=True) # B,1,3

    partial_mirror = copy.deepcopy(partial)
    partial_mirror[:,:,0:3] = partial_mirror[:,:,0:3] - center
    partial_mirror[:,:,axis] = -partial_mirror[:,:,axis]

    if partial_mirror.shape[2] >= 6:
        # in this case it contains normals from dim 3 to 5
        partial_mirror[:,:,axis+3] = -partial_mirror[:,:,axis+3]

    partial_mirror[:,:,0:3] = partial_mirror[:,:,0:3] + center
    return partial_mirror

def down_sample_points(xyz, npoints):
    # xyz is of shape (B,N,F), we assume it first 3 dimensions are 3d xyz coordinates

    xyz_flipped = xyz.transpose(1, 2).contiguous() # shape (B,F,N)
    ori_xyz = xyz[:,:,0:3].contiguous() # (B,N,3)
    idx = pointnet2_utils.furthest_point_sample(ori_xyz, npoints)
    new_xyz = pointnet2_utils.gather_operation(xyz_flipped, idx) # shape (B,F,npoints)
    new_xyz = new_xyz.transpose(1, 2).contiguous() # shape (B,npoints, F)
    # idx = torch.randperm(xyz.shape[1])[0:npoints]
    # new_xyz = xyz[:,idx,:]
    return new_xyz
    
def mirror_and_concat(partial, axis=2, num_points=[2048, 3072], attach_label=False, permute=True):
    # whether to attach 1 to real points and -1 to mirrored points
    B, N, _ = partial.size()
    partial_mirror = mirror(partial, axis=axis)

    device = partial.device
    dtype = partial.dtype
    if attach_label:
        partial = torch.cat([partial, torch.ones(B,N,1, device=device, dtype=dtype)], dim=2) # (B.N,4)
        partial_mirror = torch.cat([partial_mirror, torch.ones(B,N,1, device=device, dtype=dtype)*(-1)], dim=2) # (B.N,4)
    
    concat = torch.cat([partial, partial_mirror], dim=1) # (B,2N,4)
    concat = concat.cuda()
    if permute:
        idx = torch.randperm(concat.shape[1])
        concat = concat[:,idx,:]
    down_sampled = [concat]
    for n in num_points:
        new_xyz = down_sample_points(concat, n)
        down_sampled.append(new_xyz)
    
    return tuple(down_sampled)

def save_mirrored_points(file_name, save_name, axis=0, num_points=2048, attach_label=False):
    data = np.load(file_name)
    points = data['points']
    points = torch.from_numpy(points)
    new_points = mirror_and_concat(points, axis= axis, num_points=[num_points], attach_label=attach_label)[1]

    result_dict = {}
    for key in data._files:
        name = os.path.splitext(key)[0]
        result_dict[name] = data[name]
    result_dict['points'] = new_points.detach().cpu().numpy()
    np.savez(save_name, **result_dict)


if __name__ == '__main__':
    file_name = '../sampling_and_inference/chair_shapenet_psr_generated_data_2048_pts_epoch_1000_iter_1483999.npz'
    save_name = '../sampling_and_inference/chair_shapenet_psr_generated_data_2048_pts_epoch_1000_iter_1483999_mirrored_weight_center.npz'
    save_mirrored_points(file_name, save_name, axis=2)