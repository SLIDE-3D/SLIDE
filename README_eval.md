# Evaluation
```
cd metrics
```
We provided scripts to compute MMD, Coverage and 1NN-ACC using CD, EMD, and normal consistency (N.C.) loss, respectively.
N.C. loss is only used when the point clouds are sample from meshes, in which case we can obtain normal for each point.
The npz files specifed by sample should contain key points, and its of shape (B, npoint, 3). It could also contain key normals of shape (B, npoint, 3) if you want to use normal consistency loss to compute the metrics.
You can copy the reference set point clouds from **/mnt/petrelfs/share_data/lvzhaoyang/shapenet_psr_val_npz_files**.   

Run the following command to compute metrics for MMD, Coverage and 1NN-ACC using CD, normal consistency (N.C.) loss. 
In this case, pytorch3d is used to compute CD and normal consistency (N.C.) loss
```
srun1 python load_evaluate.py \
--sample ../pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_lamp_ae_trained_on_lamp/generated_samples/ema_kps_0.999_lat_ddpm_0.999/mesh_rc_noise_0_symmetry/shapenet_psr_generated_data_2048_pts/visualization_results_at_iteration_00000000_epoch_0000/points_sampled_from_mesh.npz \
--ref ../pointnet2/exps/shapenet_psr_validation_set/lamp_03636649_231_samples.npz \
--normalize_bdx true --pytorch3d \
--exp_id lamp_latent_ddpm_mesh_rc_noise_0_symmetry --exp_record_path eval_result --exp_record_file metrcis_cd_normal.csv
```

If you want to compute metrics using EMD distance, you need to use the cuda11.0 env:
```
conda activate cuda11.0
source setup_env_lib.sh
```
Then run 
```
srun2 python load_evaluate.py \
--sample ../pointnet2/exps/exp_shapenet_psr_generation/latent_ddpm_exps/16_keypoints/T1000_betaT0.02_shapenet_psr_latent_ddpm_ae_kp_noise_0.04_keypoint_conditional_latent_dim_16_32_lamp_ae_trained_on_lamp/generated_samples/ema_kps_0.999_lat_ddpm_0.999/mesh_rc_noise_0_symmetry/shapenet_psr_generated_data_2048_pts/visualization_results_at_iteration_00000000_epoch_0000/points_sampled_from_mesh.npz \
--ref ../pointnet2/exps/shapenet_psr_validation_set/lamp_03636649_231_samples.npz \
--normalize_bdx true \
--exp_id lamp_latent_ddpm_mesh_rc_noise_0_symmetry --exp_record_path eval_result --exp_record_file metrcis_cd_emd.csv
```