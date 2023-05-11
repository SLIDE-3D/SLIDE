# import open3d as o3d
from plyfile import PlyData, PlyElement
import argparse
import yaml
import numpy as np
import os
import pdb

def numpy_2darray_to_plyelement(a, name='points', dtype='f4'):
    # a is of shape N,D
    temp = np.empty(a.shape[0], dtype=[(name, dtype, (a.shape[1],))])     
    temp[name] = a
    temp_el = PlyElement.describe(temp, name)
    return temp_el

def visualize_pcd(pcd_file, include_idx_to_save_name=True):
    file_path, file_name = os.path.split(pcd_file)
    file_name = os.path.splitext(file_name)[0]
    save_dir = os.path.join(file_path, file_name+'_visualization')
    os.makedirs(save_dir, exist_ok=True)

    data = np.load(pcd_file)
    points = data['points']
    # pdb.set_trace()
    if 'normals.npy' in data._files:
        normals = data['normals']
        points = np.concatenate([points, normals], axis=2)
    try:
        label = data['label']
        category = data['category']
        category_name = data['category_name']
    except:
        category = None
        category_name = None
        label = np.zeros(points.shape[0]) * (-1)
    
    for idx in range(points.shape[0]):
        try:
            name = category_name[idx]
        except:
            name = 'shape'
        if include_idx_to_save_name:
            save_file_name = 'pcd_%s_label_%s_%s.xyz' % (str(idx).zfill(3), str(label[idx]).zfill(2), name)
        else:
            save_file_name = 'label_%s_%s.xyz' % (str(label[idx]).zfill(2), name)
        save_file_name = os.path.join(save_dir, save_file_name)
        np.savetxt(save_file_name, points[idx], delimiter=" ")

        if 'keypoint.npy' in data._files:
            if include_idx_to_save_name:
                save_file_name = 'pcd_%s_label_%s_%s_keypoint.xyz' % (str(idx).zfill(3), str(label[idx]).zfill(2), name)
            else:
                save_file_name = 'label_%s_%s_keypoint.xyz' % (str(label[idx]).zfill(2), name)
            save_file_name = os.path.join(save_dir, save_file_name)
            np.savetxt(save_file_name, data['keypoint'][idx], delimiter=" ")
        
        if 'gt_points.npy' in data._files:
            if include_idx_to_save_name:
                save_file_name = 'pcd_%s_label_%s_%s_gt_points.xyz' % (str(idx).zfill(3), str(label[idx]).zfill(2), name)
            else:
                save_file_name = 'label_%s_%s_gt_points.xyz' % (str(label[idx]).zfill(2), name)
            save_file_name = os.path.join(save_dir, save_file_name)
            np.savetxt(save_file_name, data['gt_points'][idx], delimiter=" ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd', type=str, default='../exps/exp_shapenet_psr_generation/T1000_betaT0.02_shapenet_psr_batchsize_64/eval_result/shapenet_psr_generated_data_2048_pts_epoch_0800_iter_383999.npz', help='the npz file that contains the point clouds')
    args = parser.parse_args()

    pcd_file = args.pcd

    visualize_pcd(pcd_file)
    
    # file_path, file_name = os.path.split(pcd_file)
    # file_name = os.path.splitext(file_name)[0]
    # save_dir = os.path.join(file_path, file_name+'_xyz_file')
    # os.makedirs(save_dir, exist_ok=True)

    # metadata_file = '../shapenet_psr_dataloader/metadata.yaml'
    # with open(metadata_file, 'r') as f:
    #     metadata = yaml.load(f, Loader=yaml.Loader)

    # categories = list(metadata.keys())
    # categories = sorted(categories)

    # all_category_names = []
    # for key in categories:
    #     all_category_names.append(metadata[key]['name'].split(',')[0])
    

    # data = np.load(pcd_file)
    # points = data['points']
    # if 'normals' in data._files:
    #     normals = data['normals']
    #     points = points
    # label = data['label']
    # category = data['category']  if 'category.npy' in data._files else None
    # category_name = data['category_name']  if 'category_name.npy' in data._files else None
    
    # # save_dir = os.path.join('../exps/point_clouds', file_name)
    # # os.makedirs(save_dir, exist_ok=True)
    # for idx in range(points.shape[0]):
    #     name = all_category_names[label[idx]] if category_name is None else category_name[idx]
    #     save_file_name = 'pcd_%s_label_%s_%s.xyz' % (str(idx).zfill(3), str(label[idx]).zfill(2), name)
    #     save_file_name = os.path.join(save_dir, save_file_name)
    #     np.savetxt(save_file_name, points[idx], delimiter=" ")