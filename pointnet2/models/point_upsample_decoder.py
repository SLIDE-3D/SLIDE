import torch
import torch.nn as nn
import pytorch3d
from pointnet2_ops.pointnet2_modules import FeatureMapModule
from models.pointnet2_feature_extractor import PointNet2Encoder
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from models.point_upsample_module import point_upsample

from data_utils.distributions import DiagonalGaussianDistribution


import copy
import numpy as np

import pdb

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)

class PointUpsampleDecoder(nn.Module):

    def __init__(self, config, in_dim, apply_kl_regularization=False):
        # in_dim is the feature dimension of the input last level point cloud
        super().__init__()
        self.hparams = config
        self.apply_kl_regularization = apply_kl_regularization
        
        # build feature extractor
        # feature extractor extracts features from current level point cloud, 
        # the features are used as queries to propagate features from the last level point cloud
        architecture = self.hparams['architecture']
        config_copy = copy.deepcopy(config)
        # output feature channel dimension need to time 2 if apply_kl_regularization
        if 'decoder_feature_dim' in architecture.keys():
            if apply_kl_regularization:
                config_copy['architecture']['decoder_feature_dim'][0] *= 2
            self.feature_extractor = PointNet2CloudCondition(config_copy)
        else:
            if apply_kl_regularization:
                config_copy['architecture']['feature_dim'][-1] *= 2
            self.feature_extractor = PointNet2Encoder(config_copy)
        

        # build feature mapper
        # feature mapper propagates features from the last level point cloud to the current level point cloud
        feature_mapper_setting = self.hparams['feature_mapper_setting']
        radius = feature_mapper_setting['radius']
        nsample = feature_mapper_setting['nsample']
        neighbor_def = feature_mapper_setting['neighbor_definition']
        activation = self.hparams.get('activation', 'relu')

        # in_dim = C1
        out_dim = feature_mapper_setting['out_dim']
        # output feature channel dimension need to time 2 if apply_kl_regularization
        if apply_kl_regularization:
            mlp_spec = [in_dim] + [out_dim*2]*feature_mapper_setting['mlp_depth']
        else:
            mlp_spec = [in_dim] + [out_dim]*feature_mapper_setting['mlp_depth']
        if 'decoder_feature_dim' in architecture.keys():
            query_feature_dim = self.hparams['architecture']['decoder_feature_dim'][0]
        else:
            query_feature_dim = self.hparams['architecture']['feature_dim'][-1]

        self.feature_mapper = FeatureMapModule(mlp_spec, radius, nsample, 
                        use_xyz=self.hparams["model.use_xyz"], include_abs_coordinate=self.hparams['include_abs_coordinate'],
                        include_center_coordinate = self.hparams.get("include_center_coordinate", False),
                        bn=self.hparams['bn'], bn_first=self.hparams["bn_first"], bias=self.hparams["bias"], 
                        res_connect=self.hparams["res_connect"],
                        first_conv=False, first_conv_in_channel=0, neighbor_def=neighbor_def,
                        activation=activation,
                        attention_setting=self.hparams['attention_setting'], 
                        query_feature_dim=query_feature_dim)

        # build upsampling module
        # upsampling module uses features at the current level point cloud to split the current level point to next level point cloud
        upsampling_setting = self.hparams['upsampling_setting']
        point_upsample_factor = upsampling_setting['point_upsample_factor']
        # if point_upsample_factor > 1:
        if upsampling_setting['first_refine_coarse_points']:
            point_upsample_factor = point_upsample_factor + 1
            if upsampling_setting['include_displacement_center_to_final_output']:
                point_upsample_factor = point_upsample_factor - 1
        else:
            assert not upsampling_setting['include_displacement_center_to_final_output']
        self.point_upsample_factor = point_upsample_factor
        self.upsampling_setting = upsampling_setting
        feature_in_dim = query_feature_dim + out_dim + self.hparams['in_fea_dim']  + 3
        points_out_dim = int(self.hparams['out_dim'] * (point_upsample_factor))

        self.fc_layer = nn.Conv1d(feature_in_dim, points_out_dim, kernel_size=1)
    
    def sample_from_distribution(self, parameters, sample_posterior):
        # parameters is of shape B,N,C
        posterior = DiagonalGaussianDistribution(parameters.transpose(1,2))
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        result = z.transpose(1,2)
        return result, posterior

    def propagate_feature(self, xyz, features, new_xyz, ts=None, label=None, sample_posterior=True):
        # xyz is of shape B,N1,3, features is of shape B,N1,C1, which are features at points xyz
        # new_xyz is of shape B, N2, C2, C2 could be 3 or a larger number, new_xyz will contain additional features if C2 > 3
        # we assume the first 3 dimensions in new_xyz are 3d coordinates 

        # we first extract features from the point cloud new_xyz as features_at_new_xyz of shape B,N2,C3
        # features_at_new_xyz are treated as queries to map features from xyz to new_xyz, 
        # and we obtain mapped_feature of shape B,N2,C4
        # mapped_feature is then concated with features_at_new_xyz to obtain the final feature for new_xyz
        # final_feature is of shape B, N2, C3+C4
        # pdb.set_trace()
        if isinstance(self.feature_extractor, PointNet2Encoder):
            out, _, _ = self.feature_extractor(new_xyz, ts=ts, label=label)
        elif isinstance(self.feature_extractor, PointNet2CloudCondition):
            out = self.feature_extractor(new_xyz, ts=ts, label=label)
        else:
            raise Exception('the type of self.feature_extractor is not supported', type(self.feature_extractor))
        # out is of shape B,N2,C3
        
        if self.apply_kl_regularization:
            # the channel dimension is divided by 2 after sampling
            out, out_posterior = self.sample_from_distribution(out, sample_posterior)

        features_at_new_xyz = out.transpose(1, 2)
        
        mapped_feature = self.feature_mapper(xyz, features.transpose(1, 2).contiguous(), new_xyz[:,:,0:3].contiguous(), subset=False, 
                                    record_neighbor_stats=False, pooling=None,
                                    features_at_new_xyz = features_at_new_xyz)
        mapped_feature = mapped_feature.transpose(1, 2)
        # mapped feature is of shape B,N2,C4
        if self.apply_kl_regularization:
            # the channel dimension is divided by 2 after sampling
            mapped_feature, mapped_feature_posterior = self.sample_from_distribution(mapped_feature, sample_posterior)

        final_feature = torch.cat([out, mapped_feature], dim=2)
        # final_feature is of shape B, N2, C3+C4

        if self.apply_kl_regularization:
            kl_loss = out_posterior.kl() + mapped_feature_posterior.kl()
        else:
            kl_loss = None
        return final_feature, kl_loss
    
    def upsample_points(self, final_feature, new_xyz):
        # we utilize the final_feature to split points in new_xyz
        # the upsampled_points will be of shape B, N2 * point_upsample_factor, self.hparams['out_dim']

        splitted_points = self.fc_layer( torch.cat([final_feature, new_xyz], dim=2).transpose(1, 2) )
        splitted_points = splitted_points.transpose(1, 2)
        # splitted_points is of shape B, N2, out_dim * point_upsample_factor

        # when new_xyz are keypoints, they may not contain normals, they may also contain additional smantic embedding
        # when split points in new_xyz, we only want to refine the 3d coordinates and 3d normals, not the additional features
        # when new_xyz do not contain normals,  in_position_and_normal_dim=3, if in this case self.hparams['out_dim']=6
        # we need to generate normals from scratch instead of refine existing normals
        if 'in_position_and_normal_dim' in self.hparams.keys():
            in_position_and_normal_dim = self.hparams['in_position_and_normal_dim']
        else:
            in_position_and_normal_dim = self.hparams['out_dim']
        coarse_points = new_xyz[:,:,0:in_position_and_normal_dim]
        if in_position_and_normal_dim < self.hparams['out_dim']:
            B,N,_ = coarse_points.shape
            pad = torch.zeros(B, N, self.hparams['out_dim']-in_position_and_normal_dim, device=coarse_points.device)
            coarse_points = torch.cat([coarse_points, pad], dim=2)
        upsampled_points = point_upsample(coarse_points, splitted_points, self.point_upsample_factor, 
                    include_displacement_center_to_final_output=self.upsampling_setting['include_displacement_center_to_final_output'],
                    output_scale_factor_value=self.upsampling_setting['output_scale_factor'], 
                    first_refine_coarse_points=self.upsampling_setting['first_refine_coarse_points'])

        num_output_points = self.upsampling_setting['num_output_points']
        assert upsampled_points.shape[1] >= num_output_points
        # pdb.set_trace()
        if upsampled_points.shape[1] > num_output_points:
            _, selected_idx = pytorch3d.ops.sample_farthest_points(upsampled_points[:,:,0:3], K=num_output_points, random_start_point=True)
            upsampled_points = pytorch3d.ops.utils.masked_gather(upsampled_points, selected_idx)
        
        return upsampled_points

    def forward(self, xyz, features, new_xyz, ts=None, label=None, sample_posterior=True):
        final_feature, kl_loss = self.propagate_feature(xyz, features, new_xyz, ts=ts, label=label, sample_posterior=sample_posterior)
        upsampled_points = self.upsample_points(final_feature, new_xyz)
        if self.apply_kl_regularization:
            return final_feature, upsampled_points, kl_loss
        else:
            return final_feature, upsampled_points   


