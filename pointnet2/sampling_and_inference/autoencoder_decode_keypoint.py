import sys
sys.path.append('../')

import pdb
import os
import json
import copy
import argparse
import numpy as np

import torch

from util import print_size
from diffusion_utils.diffusion import LatentDiffusion
from visualization_tools.visualize_pcd import visualize_pcd

from models.autoencoder import PointAutoencoder
from shapenet_psr_dataloader.npz_dataset import GeneralNpzDataset

# from autoencoder_evaluation import evaluate_per_rank
from data_utils.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict, read_json_file, autoencoder_read_config

def evaluate(net, eval_dataloader, dataset, save_dir, trainset_config, include_idx_to_save_name):
    assert dataset == 'shapenet_psr_dataset'

    net.eval()

    total_keypoints = []
    total_reconstructed_pointcloud = []
    total_generated_label = []
    total_generated_category = []
    total_generated_category_name = []
    with torch.no_grad():
        for idx, data in enumerate(eval_dataloader):
            keypoint = data['keypoint'].cuda()
            keypoint_feature = data['keypoint_feature'].cuda()
            label = data['label'].cuda()
            reconstructed_pointcloud = net.decode(keypoint.float(), keypoint_feature.float(), ts=None, label=label)
            
            total_keypoints.append(keypoint.detach().cpu().numpy())
            total_reconstructed_pointcloud.append(reconstructed_pointcloud.detach().cpu().numpy())
            total_generated_label.append(label.detach().cpu().numpy())
            total_generated_category = total_generated_category + data['category']
            total_generated_category_name = total_generated_category_name + data['category_name']
    
    total_keypoints = np.concatenate(total_keypoints, axis=0)
    total_reconstructed_pointcloud = np.concatenate(total_reconstructed_pointcloud, axis=0)
    total_generated_label = np.concatenate(total_generated_label, axis=0)
    
    result = {'points':total_reconstructed_pointcloud[:,:,0:3], 'label':total_generated_label, 'category':total_generated_category, 
            'category_name':total_generated_category_name, 'keypoint': total_keypoints}
    if total_reconstructed_pointcloud.shape[2] == 6:
        result['normals'] = total_reconstructed_pointcloud[:,:,3:6]
    pcd_save_file = os.path.join(save_dir, 'reconstructed_pcd.npz')
    np.savez(pcd_save_file, **result)
    print('reconstructed_pcd has been saved to', pcd_save_file)
    visualize_pcd(pcd_save_file, include_idx_to_save_name=include_idx_to_save_name)
    


if __name__ == '__main__':
    # this file use a autoecnoder to decode keypoint with features to a point cloud
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_airplane.json', help='JSON file for configuration')
    parser.add_argument('--ckpt', type=str, default='../exps/exp_shapenet_psr_generation/autoencoder_exps/16_keypoints/shapenet_psr_autoencoder_airplane_batchsize_32_kl_1e-5_16_kps_noise_0.04_latent_dim_16_32_normal_weight_0_0_0.1_with_augm/checkpoint/pointnet_ckpt_309749.pkl', help='the checkpoint to use')
    parser.add_argument('--dataset_path', type=str, default='../exps/shapenet_psr_validation_set/airplane_02691156_404_samples.npz', help='the npz file that stores the point clouds')
    parser.add_argument('--save_dir', type=str, default='../exps/shapenet_psr_validation_set/airplane_02691156_404_samples_autoencode_result_airplane_autoencoder', help='the directory to save point clouds')
    parser.add_argument('--batch_size', type=int, default=32, help='the batchsize to use')
    parser.add_argument('--not_include_idx_to_save_name', action='store_true', help='whether to not include idx to the save name of generated point clouds. This is used only when each point cloud has a unique category_name')
    
    args = parser.parse_args()

    '''
    srun1 python autoencoder_decode_keypoint.py --config ../configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_lamp.json --ckpt ../exps/exp_shapenet_psr_generation/autoencoder_exps/16_keypoints/shapenet_psr_autoencode_batchsize_32_kl_1e-5_16_kps_noise_0.04_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_lamp/checkpoint/pointnet_ckpt_106679.pkl  --dataset_path ../exps/controllable_generation/lamp/pcd_192_label_06_lamp/shape_combination_only_combine_top/combined_keypoint_position_and_feature.npz --save_dir ../exps/controllable_generation/lamp/pcd_192_label_06_lamp/shape_combination_only_combine_top --not_include_idx_to_save_name
    
    srun1 python autoencoder_decode_keypoint.py --config ../configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_lamp.json --ckpt ../exps/exp_shapenet_psr_generation/autoencoder_exps/16_keypoints/shapenet_psr_autoencode_batchsize_32_kl_1e-5_16_kps_noise_0.04_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_lamp/checkpoint/pointnet_ckpt_106679.pkl  --dataset_path ../exps/controllable_generation/lamp/pcd_192_label_06_lamp/shape_combination_only_combine_top/combined_keypoint_position_and_feature.npz --save_dir ../exps/controllable_generation/lamp/pcd_192_label_06_lamp/shape_combination_only_combine_top --not_include_idx_to_save_name

    srun1 python autoencoder_decode_keypoint.py --config ../configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_car.json --ckpt ../exps/exp_shapenet_psr_generation/autoencoder_exps/16_keypoints/shapenet_psr_autoencode_batchsize_32_kl_1e-5_16_kps_noise_0.04_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_car/checkpoint/pointnet_ckpt_836399.pkl --dataset_path ../exps/controllable_generation/blind_interpolation/car/004-th_pair_interpolation/keypoint_and_features.npz --save_dir ../exps/controllable_generation/blind_interpolation/car/004-th_pair_interpolation

    srun1 python autoencoder_decode_keypoint.py --config ../configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_chair.json --ckpt ../exps/exp_shapenet_psr_generation/autoencoder_exps/16_keypoints/shapenet_psr_autoencode_batchsize_32_kl_1e-5_16_kps_noise_0.04_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_chair/checkpoint/pointnet_ckpt_712319.pkl --dataset_path ../exps/controllable_generation/blind_interpolation/chair/001-th_pair_interpolation/keypoint_and_features.npz --save_dir ../exps/controllable_generation/blind_interpolation/chair/001-th_pair_interpolation
    '''

    # Parse configs. Globals nicer in this case
    global config
    config = read_json_file(args.config)
    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))
    
    global train_config
    train_config = config["train_config"]        # training parameters
    
    # global pointnet_config
    # pointnet_config = config["pointnet_config"]     # to define pointnet
    # # global diffusion_config
    # # diffusion_config = config["diffusion_config"]    # basic hyperparameters
    
    global trainset_config
    if train_config['dataset'] == 'mvp_dataset':
        trainset_config = config["mvp_dataset_config"]
    elif train_config['dataset'] == 'shapenet_psr_dataset':
        trainset_config = config['shapenet_psr_dataset_config']
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])

    # global diffusion_hyperparams 
    # diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    # global standard_diffusion_config
    # standard_diffusion_config = config['standard_diffusion_config']
    # config = read_json_file(args.config)
    config_file_path = os.path.split(args.config)[0]
    global encoder_config
    global decoder_config_list
    encoder_config, decoder_config_list = autoencoder_read_config(config_file_path, config)
       
    
    try:
        print('Using cuda device', os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
    except:
        print('CUDA_VISIBLE_DEVICES has bot been set', flush=True)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    autoencoder = PointAutoencoder(encoder_config, decoder_config_list, 
                apply_kl_regularization=config['pointnet_config'].get('apply_kl_regularization', False),
                kl_weight=config['pointnet_config'].get('kl_weight', 0))
    
    autoencoder.load_state_dict( torch.load(args.ckpt, map_location='cpu')['model_state_dict'] )
    autoencoder.cuda()
    autoencoder.eval()
    print('autoencoder size:')
    print_size(autoencoder)
    # diffusion_model = LatentDiffusion(standard_diffusion_config, autoencoder=autoencoder)

    task = config['train_config']['task']
    # trainset_config['eval_batch_size'] = args.batch_size
    trainset_config['keypoint_noise_magnitude'] = 0

    test_dataset = GeneralNpzDataset(args.dataset_path, scale=1, noise_magnitude=0, rank=0, world_size=1, 
                                        data_key='keypoint', data_key_split_names=None, data_key_split_dims=None)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    # pdb.set_trace()
    os.makedirs(args.save_dir, exist_ok=True)
    # evaluate_per_rank(autoencoder, testloader, train_config['dataset'], args.save_dir, 0, 0, trainset_config, 
    #         rank=0, world_size=1, save_reconstructed_pcd=True)
    evaluate(autoencoder, testloader, train_config['dataset'], args.save_dir, trainset_config, not args.not_include_idx_to_save_name)
    