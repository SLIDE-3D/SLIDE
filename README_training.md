# Download Training Dataset
Download the processed ShapeNet dataset from [here](https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip), unzip it and move it to the folder **pointnet2/data**.


# Train the Sparse Latent Point Diffusion Model
## Overview
We need to train 4 models in general. 
1. We need to train a Shape as Points [SAP](https://github.com/autonomousvision/shape_as_points) model that can reconstruct meshes from point clouds. 
2. We need to train a point cloud autoencoder that can encode a dense point cloud to the features at the sampled sparse latent points, and recontruct the original dense point cloud from the sparse latent points.
3. We need to train a diffusion model that learns the distribution of the positions of the sparse latent points.
4. We need to train a diffusion model that learns the distribution of the features of the sparse latent points conditioned on their positions. 

The code for training the models is in the folder **pointnet2**.
```
cd pointnet2
```

## Train the SAP Models
The SAP model first upsamples the input point cloud and then use a Differentiable Poisson Surface Reconstruction (DPSR) algorithm to reconsturct meshes from the upsampled point cloud.
Below we show the command to train the upsampling network:
```
python distributed.py --config configs/shapenet_psr_configs/refine_and_upsample_configs/config_refine_and_upsample_standard_attention_s3_noise_0.02_symmetry.json --CUDA_VISIBLE_DEVICES 0,1
```
The code uses multi-GPU for training by default, and we can use the argument ``CUDA_VISIBLE_DEVICES`` to control which GPUs to use.
It takes about 5~7 days on two NVIDIA A100 GPUs.
**train_upsampler.py** is used to train the model.
Checkpoints and evaluation results will be saved to the directory specified by the argumement ``train_config.root_directory`` and ``pointnet_config.model_name`` in the config file.
You may choose a checkpoint with the lowest dpsr_grid_L2_loss.
We provide 4 config files in **configs/shapenet_psr_configs/refine_and_upsample_configs**.
Each corresponds to a different SAP model described in Section B.9 in the appendix.

## Train Point Cloud Autoencoders
We train point cloud autoencoders for every category.
For each category, we train two autoencoders. They are diferent in how to choose the initial point when sampling the sparse latent points using farthest point sampling (FPS): One chooses the centroid of the input point cloud as the initial point, and the other randomly chooses a point from the input point cloud as the initial point.
We provide configs to train the autoencoders in the folder **configs/shapenet_psr_configs/autoencoder_configs** for 5 categories: Airplane, cabinet, car, chair, and lamp.

Run the following command to train an autoencoder (choose centroid as the initial point) for airplane:
```
python distributed.py --config configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_keypoints_latent_dim_16_32_normal_weight_0_0_0.1_with_augm_kp_noise_0.04_airplane.json --CUDA_VISIBLE_DEVICES 0,1
```
Run the following command to train an autoencoder (randomly choose a point as the initial point) for lamp:
```
python distributed.py --config configs/shapenet_psr_configs/autoencoder_configs/config_autoencoder_s3_kl_1e-5_16_kp_latent_dim_16_32_normal_weight_0.1_with_augm_kp_noise_0.04_not_add_centroid_as_first_kp_lamp.json --CUDA_VISIBLE_DEVICES 0,1
```
The code uses multi-GPU for training by default, and we can use the argument ``CUDA_VISIBLE_DEVICES`` to control which GPUs to use.
It takes about 2~3 days on two NVIDIA A100 GPUs.
It uses the file **train_autoencoder.py** to train the autoencoder. 

## Train DDPMs that Learn the Positions of the Sparse Latent Points
The config files to train the position DDPMs are in the folder **configs/shapenet_psr_configs/ddpm_keypoint_training_configs**
Run the following command to train a DDPM (choose centroid as the initial point) for airplane:
```
export CUDA_VISIBLE_DEVICES=0 && python train.py --config configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_airplane_02691156.json
```
Run the following command to train a DDPM (randomly choose a point as the initial point) for lamp:
```
export CUDA_VISIBLE_DEVICES=0 && python train.py --config configs/shapenet_psr_configs/ddpm_keypoint_training_configs/config_standard_attention_batchsize_32_s3_ema_model_keypoint_lamp_03636649_not_add_centroid_as_first_kp.json
```
The code uses a single GPU for training.
It takes about 1~3 day on a single NVIDIA A100 GPU.

## Train DDPMs that Learn the Features of the Sparse Latent Points
Next, we train the diffusion model that generate features of the sparse latent points conditioned on their positions.
The config files to train the feature DDPMs are in the folder **configs/shapenet_psr_configs/latent_ddpm_training_configs**.

Run the following command to train a DDPM (choose centroid as the initial point) for airplane:

```
export CUDA_VISIBLE_DEVICES=0 && python train_latent_ddpm.py --config configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_airplane_ae_trained_on_airplane.json
```

Run the following command to train a DDPM (randomly choose a point as the initial point) for lamp:
```
export CUDA_VISIBLE_DEVICES=0 && python train_latent_ddpm.py --config configs/shapenet_psr_configs/latent_ddpm_training_configs/config_latent_ddpm_s3_kp_noise_0.04_keypoint_conditional_not_add_centroid_as_first_kp_lamp.json
```
The code uses a single GPU for training.
It takes about 2~3 days on a single A100 GPU.
Note that in the training config file, we need to provide the information of the pretrained autoencoder in the argument **autoencoder_config**.
We need to provide the config file that we use to train the autoencoder in **config_file**, and the path to the checkpoint of the autoencoder in **ckpt**.
We have set them to pretrained checkpoints that we release, you can replace them with the autoencoder checkpoints trained by yourself.
