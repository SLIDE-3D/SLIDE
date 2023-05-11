# import open3d as o3d
# from plyfile import PlyData, PlyElement
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import yaml
import numpy as np
import os
from tqdm import tqdm
import pdb

def numpy_2darray_to_plyelement(a, name='points', dtype='f4'):
    # a is of shape N,D
    temp = np.empty(a.shape[0], dtype=[(name, dtype, (a.shape[1],))])     
    temp[name] = a
    temp_el = PlyElement.describe(temp, name)
    return temp_el

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd', type=str, default='../exps/exp_shapenet_psr_generation/T1000_betaT0.02_shapenet_psr_batchsize_64/eval_result/shapenet_psr_generated_data_2048_pts_epoch_0800_iter_383999.npz', help='the npz file that contains the point clouds')
    args = parser.parse_args()

    pcd_file = args.pcd
    # pcd_file = 'visualization_results/T1000_betaT0.02_shapenet_psr_batchsize_32_standard_diffusion_scale_loss_terms_scale_factor_1_mixed_cd_p_epsilon_mse_1.2/shapenet_psr_generated_data_2048_pts_epoch_0160_iter_153439.npz'
    
    file_path, file_name = os.path.split(pcd_file)
    file_name = os.path.splitext(file_name)[0]
    save_dir = os.path.join(file_path, file_name+'_image_file')
    os.makedirs(save_dir, exist_ok=True)

    metadata_file = '../shapenet_psr_dataloader/metadata.yaml'
    with open(metadata_file, 'r') as f:
        metadata = yaml.load(f, Loader=yaml.Loader)

    categories = list(metadata.keys())
    categories = sorted(categories)

    all_category_names = []
    for key in categories:
        all_category_names.append(metadata[key]['name'].split(',')[0])
    

    data = np.load(pcd_file)
    points = data['points']
    label = data['label']
    category = data['category']  if 'category.npy' in data._files else None
    category_name = data['category_name']  if 'category_name.npy' in data._files else None
    
    
    for idx in tqdm(range(points.shape[0])):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        x = points[idx,:,0]
        y = points[idx,:,2]
        z = points[idx,:,1]
        ax.scatter(x, y, z, c=z, cmap='jet')
        plt.grid(False)
        plt.axis('off')
        # plt.show()
        
        name = all_category_names[label[idx]] if category_name is None else category_name[idx]
        save_file_name = 'pcd_%s_label_%s_%s.png' % (str(idx).zfill(3), str(label[idx]).zfill(2), name)
        save_file_name = os.path.join(save_dir, save_file_name)
        fig.savefig(save_file_name)
        plt.close()