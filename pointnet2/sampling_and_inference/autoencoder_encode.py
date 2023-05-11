import sys
sys.path.append('../')

import pdb
import os
import json
import copy
import argparse

import torch

# from util import calc_diffusion_hyperparams
from util import print_size
from diffusion_utils.diffusion import LatentDiffusion

# from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
# from models.point_upsample_module import point_upsample
# from models.pointwise_net import get_pointwise_net

from models.autoencoder import PointAutoencoder
from shapenet_psr_dataloader.npz_dataset import GeneralNpzDataset

# from mesh_evaluation import evaluate_per_rank
from autoencoder_evaluation import evaluate_per_rank
from data_utils.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict, read_json_file, autoencoder_read_config
    


if __name__ == '__main__':
    # this file use a autoecnoder encode input pcd and visualize the reconstruction process
    # the keypoints can be fps sampled or provided in the npz file specified by dataset_path, with key keypoint
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_airplane.json', help='JSON file for configuration')
    parser.add_argument('--ckpt', type=str, default='../exps/exp_shapenet_psr_generation/autoencoder_exps/16_keypoints/shapenet_psr_autoencoder_airplane_batchsize_32_kl_1e-5_16_kps_noise_0.04_latent_dim_16_32_normal_weight_0_0_0.1_with_augm/checkpoint/pointnet_ckpt_309749.pkl', help='the checkpoint to use')
    parser.add_argument('--dataset_path', type=str, default='../exps/shapenet_psr_validation_set/airplane_02691156_404_samples.npz', help='the npz file that stores the point clouds')
    parser.add_argument('--save_dir', type=str, default='../exps/shapenet_psr_validation_set/airplane_02691156_404_samples_autoencode_result_airplane_autoencoder', help='the directory to save point clouds')
    parser.add_argument('--batch_size', type=int, default=32, help='the batchsize to use')
    parser.add_argument('--keypoint_source', type=str, default='farthest_points_sampling', help='if farthest_points_sampling, we use fps to sample keypoints, otherwise, we use keypoints stored in the npz file')
    parser.add_argument('--save_keypoint_feature', action='store_true', help='whether to save the generated features at every keypoint')
    
    args = parser.parse_args()

    '''
    srun1 python autoencoder_encode.py --config ../configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_lamp.json --ckpt ../exps/exp_shapenet_psr_generation/autoencoder_exps/16_keypoints/shapenet_psr_autoencode_batchsize_32_kl_1e-5_16_kps_noise_0.04_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_lamp/checkpoint/pointnet_ckpt_106679.pkl  --dataset_path ../exps/shapenet_psr_train_set/lamp_03636649_256_samples.npz --save_dir ../exps/shapenet_psr_train_set

    srun1 python autoencoder_encode.py --config ../configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_car.json --ckpt ../exps/exp_shapenet_psr_generation/autoencoder_exps/16_keypoints/shapenet_psr_autoencode_batchsize_32_kl_1e-5_16_kps_noise_0.04_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_car/checkpoint/pointnet_ckpt_836399.pkl  --dataset_path ../exps/controllable_generation/blind_interpolation/car/target_pcd_and_keypoint.npz --save_dir ../exps/controllable_generation/blind_interpolation/car/target_reconstruction --keypoint_source file --save_keypoint_feature

    srun1 python autoencoder_encode.py --config ../configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_chair.json --ckpt ../exps/exp_shapenet_psr_generation/autoencoder_exps/16_keypoints/shapenet_psr_autoencode_batchsize_32_kl_1e-5_16_kps_noise_0.04_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_chair/checkpoint/pointnet_ckpt_712319.pkl  --dataset_path ../exps/controllable_generation/blind_interpolation/chair/source_pcd_and_keypoint.npz --save_dir ../exps/controllable_generation/blind_interpolation/chair/source_reconstruction --keypoint_source file --save_keypoint_feature
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
                                        data_key='points', data_key_split_names=None, data_key_split_dims=None)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    # pdb.set_trace()
    os.makedirs(args.save_dir, exist_ok=True)
    evaluate_per_rank(autoencoder, testloader, train_config['dataset'], args.save_dir, 0, 0, trainset_config, 
            rank=0, world_size=1, save_reconstructed_pcd=True, keypoint_source=args.keypoint_source,
            save_keypoint_feature=args.save_keypoint_feature)
    