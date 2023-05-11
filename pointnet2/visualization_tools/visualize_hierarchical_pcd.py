# import open3d as o3d
from plyfile import PlyData, PlyElement
import argparse
import yaml
import numpy as np
import os
import pickle
import shutil
import pdb

def numpy_2darray_to_plyelement(a, name='points', dtype='f4'):
    # a is of shape N,D
    temp = np.empty(a.shape[0], dtype=[(name, dtype, (a.shape[1],))])     
    temp[name] = a
    temp_el = PlyElement.describe(temp, name)
    return temp_el

def visualize_hierarchical_pcd(pcd_file):
    file_path, file_name = os.path.split(pcd_file)
    file_name = os.path.splitext(file_name)[0]
    save_dir = os.path.join(file_path, file_name+'_visualization')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    handle = open(pcd_file, 'rb')
    data = pickle.load(handle)
    handle.close()

    points_list = data['hierarchical_pointcloud']
    label = data['label']
    category = data['category']
    category_name = data['category_name']
    gt_points = data['gt_points']
    models = data['model']

    
    for idx in range(points_list[0].shape[0]):
        # make save dir
        name = category_name[idx]
        current_save_dir = 'pcd_%s_label_%s_%s' % (str(idx).zfill(3), str(label[idx]).zfill(2), name)
        current_save_dir = os.path.join(save_dir, current_save_dir)
        os.makedirs(current_save_dir, exist_ok=True)

        # save gt points
        save_name = 'pcd_%s_label_%s_%s_gt_model_%s.xyz' % (
                    str(idx).zfill(3), str(label[idx]).zfill(2), name, models[idx])
        save_name = os.path.join(current_save_dir, save_name)
        current_gt_points = gt_points[idx]
        np.savetxt(save_name, current_gt_points, delimiter=" ")

        total_pointcloud = None
        length = current_gt_points.max(axis=0) - current_gt_points.min(axis=0)
        length = length[2]
        # visualize the point cloud hierarchy
        for i in range(len(points_list)):
            current_points = points_list[i][idx, :,:]
            if total_pointcloud is None:
                total_pointcloud = current_points[:,0:3]
            else:
                shift = np.array([[0,0,-length*1.2]]) * i
                total_pointcloud = np.concatenate([total_pointcloud, current_points[:,0:3]+shift], axis=0)
            save_name = 'pcd_%s_label_%s_%s_level_%d_points_%s.xyz' % (
                str(idx).zfill(3), str(label[idx]).zfill(2), name, i, str(current_points.shape[0]).zfill(4))
            save_name = os.path.join(current_save_dir, save_name)
            np.savetxt(save_name, current_points, delimiter=" ")

        # save all in one pointcloud
        shift = np.array([[0,0,-length*1.2]]) * (i+1)
        total_pointcloud = np.concatenate([total_pointcloud, current_gt_points[:,0:3]+shift], axis=0)
        save_name = 'pcd_%s_label_%s_%s_all_in_one.xyz' % (
            str(idx).zfill(3), str(label[idx]).zfill(2), name)
        save_name = os.path.join(current_save_dir, save_name)
        np.savetxt(save_name, total_pointcloud, delimiter=" ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd', type=str, default='../exps/exp_shapenet_psr_generation/T1000_betaT0.02_shapenet_psr_batchsize_64/eval_result/shapenet_psr_generated_data_2048_pts_epoch_0800_iter_383999.npz', help='the npz file that contains the point clouds')
    args = parser.parse_args()

    visualize_hierarchical_pcd(args.pcd)
    
    