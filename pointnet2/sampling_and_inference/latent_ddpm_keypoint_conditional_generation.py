import sys
sys.path.append('../')

import pdb
import os
import json
import copy
import argparse
import numpy as np

import torch

# from util import calc_diffusion_hyperparams
from util import print_size
from diffusion_utils.diffusion import LatentDiffusion

from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from models.point_upsample_module import point_upsample
# from models.pointwise_net import get_pointwise_net

from models.autoencoder import PointAutoencoder

from mesh_evaluation import evaluate_per_rank
from data_utils.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict, read_json_file, autoencoder_read_config


# def main(ckpt, save_dir, diffusion_model=None, rank=0, num_gpus=1):

#     net = PointNet2CloudCondition(pointnet_config).cuda()
#     checkpoint = torch.load(ckpt, map_location='cpu')
#     net.load_state_dict(checkpoint['model_state_dict'])

#     evaluate_per_rank(net, trainset_config, diffusion_hyperparams, save_dir, point_feature_dim=pointnet_config['in_fea_dim'], 
#                         diffusion_model=diffusion_model, rank=rank, world_size=num_gpus, ckpt_info='', split_points_and_normals=True)

    


if __name__ == '__main__':
    # this file generate features on given keypoints and then reconstruct the dense point clouds
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_keypoint_conditional_chair.json', help='JSON file for configuration')
    parser.add_argument('--ckpt', type=str, default='../exps/exp_shapenet_psr_generation/latent_ddpm_exps/T1000_betaT0.02_shapenet_psr_latent_ddpm_keypoint_conditional_latent_dim_16_32_chair/checkpoint/pointnet_ckpt_1483999.pkl', help='the checkpoint to use')
    parser.add_argument('--ema_idx', type=int, default=1, help='the idx of the ema state to use')
    parser.add_argument('--keypoint_file', type=str, default='../exps/controllable_generation/chair/pcd_001_label_04_chair.npz', help='the npz file that stores the keypoints, it should contain keys: points(keypoints: shape (B,N,3)), label (B), category (B), category_name (B)')
    parser.add_argument('--save_dir', type=str, default='', help='the directory to save point clouds')
    parser.add_argument('--batch_size', type=int, default=32, help='the batchsize to use')
    parser.add_argument('--local_resampling', action='store_true', help='if false, we sample features for all points in keypoint_file; if true, we resample features only for a portion of points in keypoint_file, while fix features for other points. In this case, keypoint_file should also contain keys keypoint_feature (B,N,F), and keypoint_mask (B,N) contain 0 and 1. 1 indicates points we want to resample features for.')
    parser.add_argument('--not_include_idx_to_save_name', action='store_true', help='whether to not include idx to the save name of generated point clouds. This is used only when each point cloud has a unique category_name')
    parser.add_argument('--save_keypoint_feature', action='store_true', help='whether to save the generated features at every keypoint')
    args = parser.parse_args()

    '''
    srun1 python latent_ddpm_keypoint_conditional_generation.py --config ../configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_airplane.json --ckpt ../exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane/checkpoint/pointnet_ckpt_884999.pkl --ema_idx 0 --keypoint_file ../exps/controllable_generation/airplane/wing_translation/dense_interpolation/keypoint.npz --batch_size 100 --save_dir ../exps/controllable_generation/airplane/wing_translation/dense_interpolation

    srun1 python latent_ddpm_keypoint_conditional_generation.py --config ../configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_chair_ae_trained_on_chair.json --ckpt ../exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_chair_ae_trained_on_chair/checkpoint/pointnet_ckpt_1483999.pkl --ema_idx 0 --keypoint_file ../exps/controllable_generation/chair/dense_interpolation/pcd_001_label_04_chair_keypoint.npz --batch_size 100 --save_dir ../exps/controllable_generation/chair/dense_interpolation

    srun1 python latent_ddpm_keypoint_conditional_generation.py --config ../configs/shapenet_psr_configs/latent_ddpm_training_configs/8_keypoints/config_dim_16_32_ae_noise_0.04_keypoint_cond_airplane_ae_trained_on_airplane.json --ckpt ../exps/exp_shapenet_psr_generation/latent_ddpm_exps/8_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane_ae_trained_on_airplane/checkpoint/pointnet_ckpt_884999.pkl --ema_idx 0 --keypoint_file ../exps/controllable_generation/interpolation/airplane/8_keypoints/keypoint.npz --batch_size 100 --save_dir ../exps/controllable_generation/interpolation/airplane/8_keypoints --not_include_idx_to_save_name --save_keypoint_feature

    srun1 python latent_ddpm_keypoint_conditional_generation.py --config ../configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_cabinet_ae_trained_on_cabinet.json --ckpt ../exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_cabinet_ae_trained_on_cabinet/checkpoint/pointnet_ckpt_344999.pkl --ema_idx 0 --keypoint_file ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_cabinet/generated_samples/ema_0.9999/shapenet_psr_generated_data_16_pts.npz --batch_size 400 --save_dir ../exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_cabinet_ae_trained_on_cabinet/generated_samples/ema_kps_0.9999_lat_ddpm_0.999

    srun1 python latent_ddpm_keypoint_conditional_generation.py --config ../configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_lamp_ae_trained_on_lamp.json --ckpt ../exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_lamp_ae_trained_on_lamp/checkpoint/pointnet_ckpt_507999.pkl --ema_idx 0 --keypoint_file ../exps/controllable_generation/lamp/pcd_192_label_06_lamp/scale_top/keypoint.npz --batch_size 400 --save_dir ../exps/controllable_generation/lamp/pcd_192_label_06_lamp/scale_top/keypoint_to_shape_with_features_saved --not_include_idx_to_save_name --save_keypoint_feature

    srun1 python latent_ddpm_keypoint_conditional_generation.py --config ../configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_airplane_ae_trained_on_airplane.json --ckpt ../exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane_ae_trained_on_airplane/checkpoint/pointnet_ckpt_884999.pkl --ema_idx 0 --keypoint_file ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane/generated_samples/ema_0.999/shapenet_psr_generated_data_16_pts.npz --batch_size 202 --save_dir ../exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_airplane_ae_trained_on_airplane/generated_samples/ema_kps_0.999_lat_ddpm_0.999

    srun1 python latent_ddpm_keypoint_conditional_generation.py --config ../configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_car_ae_trained_on_car.json --ckpt ../exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_car_ae_trained_on_car/checkpoint/pointnet_ckpt_1607199.pkl --ema_idx 0 --keypoint_file ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_car/generated_samples/ema_0.9999/shapenet_psr_generated_data_16_pts.npz --batch_size 202 --save_dir ../exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_car_ae_trained_on_car/generated_samples/ema_kps_0.9999_lat_ddpm_0.999
    '''

    # Parse configs. Globals nicer in this case
    global config
    config = read_json_file(args.config)
    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))
    
    global train_config
    train_config = config["train_config"]        # training parameters
    # global dist_config
    # dist_config = config["dist_config"]         # to initialize distributed training
    # if len(args.dist_url) > 0:
    #     dist_config['dist_url'] = args.dist_url
    global pointnet_config
    pointnet_config = config["pointnet_config"]     # to define pointnet
    # global diffusion_config
    # diffusion_config = config["diffusion_config"]    # basic hyperparameters
    
    global trainset_config
    if train_config['dataset'] == 'mvp_dataset':
        trainset_config = config["mvp_dataset_config"]
    elif train_config['dataset'] == 'shapenet_psr_dataset':
        trainset_config = config['shapenet_psr_dataset_config']
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])

    # global diffusion_hyperparams 
    # diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    global standard_diffusion_config
    standard_diffusion_config = config['standard_diffusion_config']

    # read autoencoder configs
    autoencoder_config_file = '../' + config['autoencoder_config']['config_file']
    global autoencoder_config
    autoencoder_config = read_json_file(autoencoder_config_file)
    autoencoder_config_file_path = os.path.split(autoencoder_config_file)[0]
    global encoder_config
    global decoder_config_list
    encoder_config, decoder_config_list = autoencoder_read_config(autoencoder_config_file_path, autoencoder_config)
    print('The autoencoder configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(autoencoder_config)), indent=4))
       
    
    try:
        print('Using cuda device', os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
    except:
        print('CUDA_VISIBLE_DEVICES has bot been set', flush=True)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    network_type = pointnet_config.get('network_type', 'pointnet++')
    assert network_type in ['pointnet++', 'pointwise_net', 'pvd']
    if network_type == 'pointnet++':
        net = PointNet2CloudCondition(pointnet_config)
    elif network_type == 'pointwise_net':
        net = get_pointwise_net(pointnet_config['network_args'])
    elif network_type == 'pvd':
        net = PVCNN2(**pointnet_config['network_args'])
    state = torch.load(args.ckpt, map_location='cpu')
    
    net.load_state_dict(state['model_state_dict'])
    if args.ema_idx >= 0:
        net.load_state_dict(state['ema_state_list'][args.ema_idx])
    net.cuda()
    net.eval()
    print('latent ddpm size:')
    print_size(net)

    autoencoder = PointAutoencoder(encoder_config, decoder_config_list, 
                apply_kl_regularization=autoencoder_config['pointnet_config'].get('apply_kl_regularization', False),
                kl_weight=autoencoder_config['pointnet_config'].get('kl_weight', 0))
    # check if the path of ckpt is absolute path
    if not os.path.isabs(config['autoencoder_config']['ckpt']):
        config['autoencoder_config']['ckpt'] = os.path.join('../', config['autoencoder_config']['ckpt'])
    else:
        config['autoencoder_config']['ckpt'] = config['autoencoder_config']['ckpt']
    autoencoder.load_state_dict( torch.load(config['autoencoder_config']['ckpt'], map_location='cpu')['model_state_dict'] )
    autoencoder.cuda()
    autoencoder.eval()
    print('autoencoder size:')
    print_size(autoencoder)
    diffusion_model = LatentDiffusion(standard_diffusion_config, autoencoder=autoencoder)

    task = config['train_config']['task']
    trainset_config['eval_batch_size'] = args.batch_size
    if len(args.save_dir) == 0:
        save_dir = os.path.split(args.keypoint_file)[0]
    else:
        save_dir = args.save_dir

    if args.local_resampling:
        data = np.load(args.keypoint_file)
        keypoint_feature = torch.from_numpy(data['keypoint_feature']).float() # (B,N,F)
        keypoint_mask = torch.from_numpy(data['keypoint_mask']).float() # (B,N)
        keypoint = torch.from_numpy(data['points']).float() # (B,N,3)
        complete_x0 = torch.cat([keypoint, keypoint_feature], dim=2) # (B,N,3+F)
    else:
        complete_x0 = None
        keypoint_mask = None

    evaluate_per_rank(net, trainset_config, None, save_dir, task,
                    point_feature_dim=pointnet_config['in_fea_dim'], 
                    rank=0, world_size=1, ckpt_info='',
                    diffusion_model=diffusion_model, keypoint_dim=3,
                    test_external_keypoint=True, external_keypoint_file=args.keypoint_file,
                    split_points_and_normals=True, include_idx_to_save_name=not args.not_include_idx_to_save_name,
                    save_keypoint_feature=args.save_keypoint_feature,
                    local_resampling=args.local_resampling, complete_x0=complete_x0, keypoint_mask=keypoint_mask)

    