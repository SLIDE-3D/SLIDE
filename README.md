# SLIDE: Controllable Mesh Generation Through Sparse Latent Point Diffusion Models
This repo intends to release code for our work: 

Zhaoyang Lyu\*, Jinyi Wang\*, Yuwei An, Ya Zhang, Dahua Lin, Bo Dai, ["Controllable Mesh Generation Through Sparse Latent Point Diffusion Models"](https://arxiv.org/abs/2303.07938).

\* Equal contribution

Project papge is [here.](https://slide-3d.github.io/)

**S**parse **L**atent po**I**nt **D**iffusion mod**E**l (**SLIDE**) is a mesh generative model based on latent diffusion models.
SLIDE encodes a 3D shape to a sparse set of latent points with features, and train diffusion models in this latent space.
Samples generated in this latent space are decoded back to point clouds by a neural decoder, and then meshes can be recontructed from the point clouds through [SAP](https://arxiv.org/abs/2106.03452). Below is an overview of the framework of SLIDE:

![slide_overview](figures/framework.png)


# Set up the environment
The code is tested with Pytorch 1.12.1, CUDA 11.3, and gcc 5.4.0, 7.3.1, or 9.4.0 (Other gcc verions may work as well, but we have not tested them.).
Run the following command to setup the environment:
```
conda env create -f environment.yml
conda activate slide
cd pointnet2_ops_lib
pip install -e .
cd ..
pip install -e .
```

# Download the Checkpoints
Download the pretrained checkpoints from [here](https://drive.google.com/file/d/16ajacjUK54T8Hai4X7O9tU7r4zgSTvpU/view?usp=share_link).
Unzip the file to the folder **pointnet2/exps**.
## Checkpoints Overview
There are 4 types of models in general.
1. Shape as Points [SAP](https://github.com/autonomousvision/shape_as_points) models that reconstruct meshes from a point cloud. 
2. Point cloud autoencoders that encode a dense point cloud to the features at the sampled sparse latent points, and recontruct the original dense point cloud from the sparse latent points.
3. Position DDPMs that learn the distribution of the positions of the sparse latent points.
4. Feature DDPMs that learn the distribution of the features of the sparse latent points conditioned on their positions. 

## SAP Models
SAP models are in the directory **mesh_overall_ckpts_and_generation_results/SAP_models**.
There are 4 SAP models in total. Each corresponds to a different SAP model described in Section B.9 in the appendix.
| Model Index | Use Symmetry | Use Normal | Add Noise | Path |
| :--------------:| :------:     | :------:   | :------:  | :------:      |
| SAP-1           | True         | True       | True      | normal_symmetry_noise_0.02 |
| SAP-2           | True         | True       | False     | normal_symmetry |
| SAP-3           | True         | False      | False     | symmetry |
| SAP-4           | False        | False      | False     | nothing |

The SAP models are trained on the 13 categories of the ShapeNet dataset.
We recommend to use SAP-1 and SAP-2 to reconstruct meshes from point clouds for best visual quality.
SAP-1 is more suitable for point clouds with defects, but may result in over-smooth surfaces.
SAP-2 reconstructs meshes more faithful to the original point cloud, but may generate less visual-appealling meshes when the input point cloud has defects.
You may choose one according to the quality of the input point cloud.

## Point Cloud Autoencoder
Autoencoder models are in the directory **mesh_overall_ckpts_and_generation_results/autoencoder_models**.
There are two subfolders **add_centroid_as_first_keypoint** and **not_add_centroid_as_first_keypoint**.
The first one stores checkpoints where centroid is set as the initial point for FPS during training.
The second one stores checkpoints where a random point is set as the initial point for FPS during training.
There are checkpoints for 5 categories (Airplane, Cabinet, Car, Chair, Lamp) in each of the two subfolders. 

## DDPMs
The position DDPMs and feature DDPMs are stored in the directories **mesh_overall_ckpts_and_generation_results/latent_position_ddpm_models** and **mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models**, respectively.
They have similar folder structures as the autoencoder models.


# Generate Meshes with the Checkpoints
## Overview
With the SAP model, the point cloud autoencoder, and the DDPMs that generate positions and features of the sparse latent points,
we can use them to generate point clouds and meshes.
First, we use the position DDPM to generate positions of the sparse latent points.
Second, we use the feature DDPM to generate features of the sparse latent points, and then recontruct point clouds from the sparse latent points.
Finally, we use the SAP model to recontruct meshes from the generated point clouds.

The code for point cloud and mesh generation is in the folder **pointnet2/sampling_and_inference**.
```
cd pointnet2/sampling_and_inference
```

## Generate Positions of the Sparse Latent Points
To generate positions of the sparse latent points, we need to provide the config file that used to train the position DDPM, the path to the saved checkpoint, and the directory to save the generated positions.

For example, run the following command to generate positions of 400 groups of sparse latent points for airplane (choose centroid as the initial point):
```
export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config ../exps/mesh_overall_ckpts_and_generation_results/latent_position_ddpm_models/add_centroid_to_first_keypoint/airplane/config_standard_attention_batchsize_32_s3_ema_model_keypoint_airplane_02691156.json \
--ckpt ../exps/mesh_overall_ckpts_and_generation_results/latent_position_ddpm_models/add_centroid_to_first_keypoint/airplane/pointnet_ckpt_884999.pkl \
--ema_idx 0 --num_samples 400 --batch_size 200 \
--save_dir ../exps/generated_point_cloud_and_mesh/airplane/centroid
```
Positions will be saved to **../exps/generated_point_cloud_and_mesh/airplane/centroid/shapenet_psr_generated_data_16_pts.npz**.
The argument ``ema_idx`` specifies the index of the ema rate of the state dict to use stored in the config file (``train_config.ema_rate``). 

Run the following command to generate positions of 400 groups of sparse latent points for lamp (randomly choose a point as the initial point):
```
export CUDA_VISIBLE_DEVICES=0 && python point_cloud_generation.py --config ../exps/mesh_overall_ckpts_and_generation_results/latent_position_ddpm_models/not_add_centroid_to_first_keypoint/lamp/config_standard_attention_batchsize_32_s3_ema_model_keypoint_lamp_03636649_not_add_centroid_as_first_kp.json \
--ckpt ../exps/mesh_overall_ckpts_and_generation_results/latent_position_ddpm_models/not_add_centroid_to_first_keypoint/lamp/pointnet_ckpt_507999.pkl \
--ema_idx 0 --num_samples 400 --batch_size 200 \
--save_dir ../exps/generated_point_cloud_and_mesh/lamp/random
```
Positions will be saved to **../exps/generated_point_cloud_and_mesh/lamp/random/shapenet_psr_generated_data_16_pts.npz**.


## Generate Features of the Sparse Latent Points and Reconstruct Point Clouds
The next step is to generate features conditioned on the positions of the sparse latent points, and reconstruct point clouds from the sparse latent points using the point cloud autoencoder.
We need to provide the config file that used to train the DDPM that generate features conditioned on the positions of the latent points, the path to the saved checkpoint, and the npz file that stores the positions of the generated latent points in the last step, and to directory to save the generated point clouds.

For example, run the following command to generate features of the 400 groups of sparse latent points for airplane and reconstruct point clouds:
```
export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config ../exps/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/airplane/config_latent_ddpm_s3_dim_16_32_ae_kp_noise_0.04_keypoint_conditional_airplane_ae_trained_on_airplane.json \
--ckpt ../exps/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/add_centroid_as_first_keypoint/airplane/pointnet_ckpt_884999.pkl \
--ema_idx 0 --batch_size 200 \
--keypoint_file ../exps/generated_point_cloud_and_mesh/airplane/centroid/shapenet_psr_generated_data_16_pts.npz \
--save_dir ../exps/generated_point_cloud_and_mesh/airplane/centroid
```
Reconstructed point clouds will be saved to  **../exps/generated_point_cloud_and_mesh/airplane/centroid/shapenet_psr_generated_data_2048_pts.npz**.

Run the following command to generate features of the 400 groups of sparse latent points for lamp and reconstruct point clouds:
```
export CUDA_VISIBLE_DEVICES=0 && python latent_ddpm_keypoint_conditional_generation.py --config ../exps/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/not_add_centroid_as_first_keypoint/lamp/config_latent_ddpm_s3_kp_noise_0.04_keypoint_conditional_not_add_centroid_as_first_kp_lamp.json \
--ckpt ../exps/mesh_overall_ckpts_and_generation_results/latent_feature_ddpm_models/not_add_centroid_as_first_keypoint/lamp/pointnet_ckpt_507999.pkl \
--ema_idx 0 --batch_size 200 \
--keypoint_file ../exps/generated_point_cloud_and_mesh/lamp/random/shapenet_psr_generated_data_16_pts.npz \
--save_dir ../exps/generated_point_cloud_and_mesh/lamp/random
```
Reconstructed point clouds will be saved to  **../exps/generated_point_cloud_and_mesh/lamp/random/shapenet_psr_generated_data_2048_pts.npz**.

## Recontruct Meshes from Generated Point Clouds
To reconstruct meshes from the generated point clouds using SAP, we need to provide the config file used to train the SAP model, the path to the saved checkpoint, the path to the generated point clouds, and directory to save reconstructed meshes. 


For example, run the following command to reconstruct meshes from the generated airplane point clouds in the last step:
```
export CUDA_VISIBLE_DEVICES=0 && python mesh_reconstruction.py \
--config ../exps/mesh_overall_ckpts_and_generation_results/SAP_models/normal_symmetry_noise_0.02/config_refine_and_upsample_standard_attention_s3_noise_0.02_symmetry.json \
--ckpt ../exps/mesh_overall_ckpts_and_generation_results/SAP_models/normal_symmetry_noise_0.02/pointnet_ckpt_901459.pkl \
--dataset_path ../exps/generated_point_cloud_and_mesh/airplane/centroid/shapenet_psr_generated_data_2048_pts.npz \
--save_dir ../exps/generated_point_cloud_and_mesh/airplane/centroid/mesh_reconstruction
```

Run the following command to reconstruct meshes from the generated lamp point clouds in the last step:
```
export CUDA_VISIBLE_DEVICES=0 && python mesh_reconstruction.py \
--config ../exps/mesh_overall_ckpts_and_generation_results/SAP_models/normal_symmetry_noise_0.02/config_refine_and_upsample_standard_attention_s3_noise_0.02_symmetry.json \
--ckpt ../exps/mesh_overall_ckpts_and_generation_results/SAP_models/normal_symmetry_noise_0.02/pointnet_ckpt_901459.pkl \
--dataset_path ../exps/generated_point_cloud_and_mesh/lamp/random/shapenet_psr_generated_data_2048_pts.npz \
--save_dir ../exps/generated_point_cloud_and_mesh/lamp/random/mesh_reconstruction
```

# Train the Models by Yourself
See README_training.md for instructions.
