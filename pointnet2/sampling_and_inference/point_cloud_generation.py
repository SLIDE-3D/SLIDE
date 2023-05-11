import sys
sys.path.append('../')

import pdb
import os
import json
import copy
import argparse

import torch

from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from shapenet_psr_dataloader.dummy_shapenet_psr_dataset import DummyShapes3dDataset

from util import calc_diffusion_hyperparams
from diffusion_utils.diffusion import Diffusion

from mesh_evaluation import evaluate_per_rank
from data_utils.json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict

def main(ckpt, ema_idx, save_dir, diffusion_model=None, rank=0, num_gpus=1):

    net = PointNet2CloudCondition(pointnet_config)
    checkpoint = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    if ema_idx >= 0:
        net.load_state_dict(checkpoint['ema_state_list'][ema_idx])
    net.cuda()

    evaluate_per_rank(net, trainset_config, diffusion_hyperparams, save_dir, train_config['task'], 
                        point_feature_dim=pointnet_config['in_fea_dim'], diffusion_model=diffusion_model, 
                        rank=rank, world_size=num_gpus, ckpt_info='', split_points_and_normals=True)
    # evaluate_per_rank(net, trainset_config, diffusion_hyperparams, save_dir, task, point_feature_dim=3, diffusion_model=None, 
    #                     rank=0, world_size=1, ckpt_info='', keypoint_dim=3, test_external_keypoint=False, external_keypoint_file=None, 
    #                     split_points_and_normals=False)

    


if __name__ == '__main__':
    # use a ddpm trained on dense point clouds to generate point clouds
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/shapenet_psr_configs/config_standard_attention_batchsize_32_s3.json', help='JSON file for configuration')
    parser.add_argument('--ckpt', type=str, default='../exps/exp_shapenet_psr_generation/T1000_betaT0.02_shapenet_psr_batchsize_32/checkpoint/pointnet_ckpt_958999.pkl', help='the checkpoint to use')
    parser.add_argument('--ema_idx', type=int, default=1, help='The idx of the ema state to use. Set it to -1 if you do not want to use ema recorded state dict of the network.')
    parser.add_argument('--num_samples', type=int, default=32, help='num samples to generate')
    parser.add_argument('--batch_size', type=int, default=32, help='the batchsize to use')
    parser.add_argument('--save_dir', type=str, default='ddpm_generated_point_clouds', help='the directory to save meshes')

    parser.add_argument('--data_clamp_range', type=float, default=1, help='the range to clamp x0, if it < 0, we will not clamp the data')
    parser.add_argument('--model_var_type', type=str, default='fixedsmall', help='it could be fixedsmall or fixedlarge')
    args = parser.parse_args()

    '''
    srun1 python point_cloud_generation.py --config ../configs/shapenet_psr_configs/ddpm_training_configs/no_normal_configs/config_standard_attention_batchsize_32_s3_ema_centered_to_centroid_airplane_02691156_no_normal.json --ckpt ../exps/exp_shapenet_psr_generation/ddpm_training_exps/no_normal_exps/T1000_betaT0.02_shapenet_psr_batchsize_32_with_ema_centered_to_centroid_airplane_02691156_no_normal/checkpoint/pointnet_ckpt_884999.pkl --ema_idx 0 --num_samples 404 --batch_size 64 --save_dir ../exps/exp_shapenet_psr_generation/ddpm_training_exps/no_normal_exps/T1000_betaT0.02_shapenet_psr_batchsize_32_with_ema_centered_to_centroid_airplane_02691156_no_normal/generated_samples/ema_0.999

    srun1 python point_cloud_generation.py --config ../configs/shapenet_psr_configs/ddpm_training_configs/config_standard_attention_batchsize_32_s3_ema_cabinet_02933112.json --ckpt ../exps/exp_shapenet_psr_generation/ddpm_training_exps/T1000_betaT0.02_shapenet_psr_batchsize_32_with_ema_cabinet_02933112/checkpoint/pointnet_ckpt_344999.pkl --ema_idx 0 --num_samples 157 --batch_size 64 --save_dir ../exps/exp_shapenet_psr_generation/ddpm_training_exps/T1000_betaT0.02_shapenet_psr_batchsize_32_with_ema_cabinet_02933112/generated_samples/ema_0.999

    srun1 python point_cloud_generation.py --config ../configs/shapenet_psr_configs/ddpm_training_configs/config_standard_attention_batchsize_32_s3_ema_car_02958343.json --ckpt ../exps/exp_shapenet_psr_generation/ddpm_training_exps/T1000_betaT0.02_shapenet_psr_batchsize_32_with_ema_car_02958343/checkpoint/pointnet_ckpt_1639999.pkl --ema_idx 0 --num_samples 749 --batch_size 64 --save_dir ../exps/exp_shapenet_psr_generation/ddpm_training_exps/T1000_betaT0.02_shapenet_psr_batchsize_32_with_ema_car_02958343/generated_samples/ema_0.999

    srun1 python point_cloud_generation.py --config ../configs/shapenet_psr_configs/ddpm_training_configs/config_standard_attention_batchsize_32_s3_ema_chair_03001627.json --ckpt ../exps/exp_shapenet_psr_generation/ddpm_training_exps/T1000_betaT0.02_shapenet_psr_batchsize_32_with_ema_chair_03001627/checkpoint/pointnet_ckpt_1483999.pkl --ema_idx 0 --num_samples 677 --batch_size 64 --save_dir ../exps/exp_shapenet_psr_generation/ddpm_training_exps/T1000_betaT0.02_shapenet_psr_batchsize_32_with_ema_chair_03001627/generated_samples/ema_0.999

    srun1 python point_cloud_generation.py --config ../configs/shapenet_psr_configs/ddpm_training_configs/config_standard_attention_batchsize_32_s3_ema_lamp_03636649.json --ckpt ../exps/exp_shapenet_psr_generation/ddpm_training_exps/T1000_betaT0.02_shapenet_psr_batchsize_32_with_ema_lamp_03636649/checkpoint/pointnet_ckpt_507999.pkl --ema_idx 0 --num_samples 231 --batch_size 64 --save_dir ../exps/exp_shapenet_psr_generation/ddpm_training_exps/T1000_betaT0.02_shapenet_psr_batchsize_32_with_ema_lamp_03636649/generated_samples/ema_0.999

    srun1 python point_cloud_generation.py --config ../configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_airplane_02691156_32_keypoints.json --ckpt ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane_32_keypoints/checkpoint/pointnet_ckpt_884999.pkl --ema_idx 1 --num_samples 404 --batch_size 202 --save_dir ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane_32_keypoints/generated_samples/ema_0.9999

    srun1 python point_cloud_generation.py --config ../configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_airplane_02691156.json --ckpt ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane/checkpoint/pointnet_ckpt_884999.pkl --ema_idx 1 --num_samples 404 --batch_size 404 --save_dir ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_airplane/generated_samples/ema_0.9999

    srun1 python point_cloud_generation.py --config ../configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_cabinet_02933112.json --ckpt ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_cabinet/checkpoint/pointnet_ckpt_344999.pkl --ema_idx 0 --num_samples 157 --batch_size 157 --save_dir ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_cabinet/generated_samples/ema_0.999

    srun1 python point_cloud_generation.py --config ../configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_car_02958343.json --ckpt ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_car/checkpoint/pointnet_ckpt_1639999.pkl --ema_idx 1 --num_samples 749 --batch_size 380 --save_dir ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_car/generated_samples/ema_0.9999

    srun1 python point_cloud_generation.py --config ../configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_chair_03001627.json --ckpt ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_chair/checkpoint/pointnet_ckpt_1483999.pkl --ema_idx 0 --num_samples 677 --batch_size 340 --save_dir ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_chair/generated_samples/ema_0.999

    srun1 python point_cloud_generation.py --config ../configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_lamp_03636649.json --ckpt ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_lamp/checkpoint/pointnet_ckpt_507999.pkl --ema_idx 1 --num_samples 231 --batch_size 116 --save_dir ../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_lamp/generated_samples/ema_0.9999
    '''

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    
    config = restore_string_to_list_in_a_dict(config)
    
    global train_config
    train_config = config["train_config"]        # training parameters
    global pointnet_config
    pointnet_config = config["pointnet_config"]     # to define pointnet
    
    global trainset_config
    if train_config['dataset'] == 'mvp_dataset':
        trainset_config = config["mvp_dataset_config"]
    elif train_config['dataset'] == 'shapenet_psr_dataset':
        trainset_config = config['shapenet_psr_dataset_config']
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])
    trainset_config['num_samples_tested'] = args.num_samples
    trainset_config['eval_batch_size'] = args.batch_size
    if not trainset_config['data_dir'].startswith('s3://'):
        trainset_config['data_dir'] = '../' + trainset_config['data_dir']

    global diffusion_config
    global diffusion_hyperparams 
    diffusion_config = None
    diffusion_hyperparams = None
    if 'diffusion_config' in config.keys():
        diffusion_config = config["diffusion_config"]    # basic hyperparameters
        diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    diffusion_model = None
    if "standard_diffusion_config" in config.keys():
        standard_diffusion_config = config["standard_diffusion_config"]
        standard_diffusion_config['model_var_type'] = args.model_var_type
        standard_diffusion_config['data_clamp_range'] = args.data_clamp_range
        diffusion_model = Diffusion(standard_diffusion_config)
    # pdb.set_trace()

    save_dir = args.save_dir
    # file_name = os.path.split(args.ckpt)[1]
    # file_name = os.path.splitext(file_name)[0]
    # model_name = args.ckpt.split('/')[-3]
    # save_dir = os.path.join(save_dir, model_name, file_name)

    try:
        print('Using cuda device', os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
    except:
        print('CUDA_VISIBLE_DEVICES has bot been set', flush=True)

    main(args.ckpt, args.ema_idx, save_dir, diffusion_model=diffusion_model, rank=0, num_gpus=1)

    