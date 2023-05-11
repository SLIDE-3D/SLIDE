import open3d as o3d
import yaml
import numpy as np
import os
import pdb

if __name__ == '__main__':
    # pcd_file = '../exps/exp_shapenet_psr_generation/T1000_betaT0.02_shapenet_psr_airplane_02691156/eval_result/shapenet_psr_generated_data_2048_pts_epoch_2900_iter_258099.npz'
    pcd_file = '../exps/exp_shapenet_psr_generation/T1000_betaT0.02_shapenet_psr/eval_result/shapenet_psr_generated_data_2048_pts_epoch_0800_iter_767199.npz'
    

    dataset_folder = '../shapenet_psr_dataloader/shapenet_psr'
    categories = os.listdir(dataset_folder)
    categories = [c for c in categories if os.path.isdir(os.path.join(dataset_folder, c))]

    sorted_categories = sorted(categories)
    sorted_categories_dict = {} # category to label number
    for i in range(len(sorted_categories)):
        sorted_categories_dict[sorted_categories[i]] = i

    data = np.load(pcd_file)
    points = data['points']
    label = data['label']
    new_label = [sorted_categories_dict[categories[l]]  for l in label]
    new_label = np.array(new_label)

    save_file = os.path.split(pcd_file)[1]
    np.savez(save_file, points=points, label=label)