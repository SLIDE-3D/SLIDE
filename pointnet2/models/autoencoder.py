import torch
import torch.nn as nn
import pytorch3d
from models.keypoint_decoder import KeypointDecoder
from models.pointnet2_feature_extractor import PointNet2Encoder
from metrics_point_cloud.chamfer_and_f1 import calc_cd
from models.point_upsample_decoder import PointUpsampleDecoder

import pdb

class PointAutoencoder(nn.Module):

    def __init__(self, encoder_config, decoder_config_list, apply_kl_regularization=False, kl_weight=0, feature_weight=None):
        super().__init__()
        self.apply_kl_regularization = apply_kl_regularization
        self.kl_weight = kl_weight
        self.feature_weight = feature_weight
        
        # build the encoder, extract features from the input point cloud
        self.encoder = PointNet2Encoder(encoder_config)

        # build keypoint_encoder, which maps the last level feature from the encoder to user specified keypoints
        feature_dim = encoder_config['architecture']['feature_dim'][-1]
        self.keypoint_encoder = PointUpsampleDecoder(decoder_config_list[0], in_dim=feature_dim, apply_kl_regularization=apply_kl_regularization)

        # build decoder, which reconstructs the input point cloud from features at keypoints
        # feature_dim = decoder_config_list[0]['architecture']['feature_dim'][-1] + decoder_config_list[0]['feature_mapper_setting']['out_dim']
        # # we already assume that decoder_config_list[0] contains a PointNet2Encoder to extract features
        # # not a PointNet2CloudCondition to extract features
        if 'decoder_feature_dim' in decoder_config_list[0]['architecture'].keys():
            feature_dim = decoder_config_list[0]['architecture']['decoder_feature_dim'][0] + decoder_config_list[0]['feature_mapper_setting']['out_dim']
        else:
            feature_dim = decoder_config_list[0]['architecture']['feature_dim'][-1] + decoder_config_list[0]['feature_mapper_setting']['out_dim']
                
        self.decoder = KeypointDecoder(decoder_config_list[1:], feature_dim)

    def encode(self, pointcloud, keypoint, ts=None, label=None, sample_posterior=True):
        out, l_xyz_encoder, _ = self.encoder(pointcloud, ts=ts, label=label)
        feature_at_keypoint, _ = self.keypoint_encoder.propagate_feature(l_xyz_encoder[-1], out, keypoint, ts=ts, label=label, sample_posterior=sample_posterior)
        return feature_at_keypoint
    
    def decode(self, keypoint, feature_at_keypoint, ts=None, label=None):
        new_xyz = self.keypoint_encoder.upsample_points(feature_at_keypoint, keypoint)
        l_xyz_decoder = self.decoder(keypoint[:,:,0:3], feature_at_keypoint, new_xyz, ts=ts, label=label)
        return l_xyz_decoder[-1]


    def forward(self, pointcloud, keypoint, ts=None, label=None, loss_type='cd_p', sample_posterior=True, 
                    return_keypoint_feature=False):
        out, l_xyz_encoder, _ = self.encoder(pointcloud, ts=ts, label=label)
        # out is of shape B,N,C, which are last level features extracted from the encoder
        feature_at_keypoint, new_xyz, kl_loss = self.keypoint_encoder(l_xyz_encoder[-1], out, keypoint, ts=ts, label=label, 
                                            sample_posterior=sample_posterior)
        
        l_xyz_decoder = self.decoder(keypoint[:,:,0:3], feature_at_keypoint, new_xyz, ts=ts, label=label)
        # the first pointcloud in l_xyz_decoder is keypoints
        # beginning from ther second point cloud, are decoder generated multi level point clouds
        # l_xyz_decoder = self.decoder(l_xyz_encoder[-1], out, keypoint, ts=None, label=label)

        assert pointcloud.shape[2] in [3,6]
        xyz = pointcloud[:,:,0:3]
        loss_list = []
        for i in range(1, len(l_xyz_decoder)):
            # the first point cloud in l_xyz_decoder do not need supervision, because it is user provided keypoints
            uvw = l_xyz_decoder[i]
            num_points = uvw.shape[1]
            _, selected_idx = pytorch3d.ops.sample_farthest_points(xyz, K=num_points, random_start_point=True)
            downsampled_pointcloud = pytorch3d.ops.utils.masked_gather(pointcloud, selected_idx)
            loss_dict = calc_cd(uvw, downsampled_pointcloud, calc_f1=True, f1_threshold=0.0001, 
                        normal_loss_type='mse')
            feature_weight = 0 if self.feature_weight is None else self.feature_weight[i-1]
            if loss_type == 'cd_p':
                loss = loss_dict['cd_p'] + loss_dict['cd_feature_p']*feature_weight
            elif loss_type == 'cd_t':
                loss = loss_dict['cd_t'] + loss_dict['cd_feature_t']*feature_weight
            else:
                raise Exception('loss type %s is not supported yet' % loss_type)
                
            if self.apply_kl_regularization and self.kl_weight > 0:
                # the kl loss only need to be added once, we add it at the last layer
                if i == len(l_xyz_decoder)-1:
                    loss_dict['kl_loss'] = kl_loss
                    loss = loss + self.kl_weight * loss_dict['kl_loss']
                else:
                    loss_dict['kl_loss'] = torch.zeros_like(loss)
            loss_dict['training_loss'] = loss
            loss_list.append(loss_dict)
        
        if return_keypoint_feature:
            return l_xyz_decoder, loss_list, feature_at_keypoint
        else:
            return l_xyz_decoder, loss_list


