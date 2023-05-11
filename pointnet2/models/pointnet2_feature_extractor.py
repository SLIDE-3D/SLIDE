import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, FeatureMapModule
from pointnet2_ops.pointnet2_utils import QueryAndGroup
from torch.utils.data import DataLoader

# from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG, calc_t_emb, swish
# from pointnet2.models.pnet import Pnet2Stage
# from pointnet2.models.model_utils import get_embedder

from models.pointnet2_ssg_sem import PointNet2SemSegSSG, calc_t_emb, swish
from models.pnet import Pnet2Stage
from models.model_utils import get_embedder

import copy
import numpy as np

import pdb

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)

class PointNet2Encoder(PointNet2SemSegSSG):

    def _build_model(self):

        self.attention_setting = self.hparams.get("attention_setting", None)
        self.global_attention_setting = self.hparams.get('global_attention_setting', None)

        self.bn = self.hparams.get("bn", True) # bn here refers to group norm
        self.scale_factor = 1 # has no actual use, we can ignore this parameter
        self.record_neighbor_stats = self.hparams["record_neighbor_stats"]
        if self.hparams["include_class_condition"]:
            # utilize the class information of the partial point cloud
            self.class_emb = nn.Embedding(self.hparams["num_class"], self.hparams["class_condition_dim"])
        
        in_fea_dim = self.hparams['in_fea_dim']
        self.attach_position_to_input_feature = self.hparams['attach_position_to_input_feature']
        if self.attach_position_to_input_feature:
            in_fea_dim = in_fea_dim + 3
        
        self.use_position_encoding = self.hparams.get('use_position_encoding', False)
        # do not use positional encoding by default, we observe that it does not help
        if self.use_position_encoding:
            multires = self.hparams['position_encoding_multires']
            self.pos_encode, pos_encode_out_dim = get_embedder(multires)
            in_fea_dim = in_fea_dim + pos_encode_out_dim

        self.in_fea_dim = in_fea_dim

        self.include_abs_coordinate = self.hparams['include_abs_coordinate']
        self.pooling = self.hparams.get('pooling', 'max')
        # pooling should be max, avg or avg_max
        # pooling will have no effect and will not be used if self.attention_setting.use_attention_module
        # we will use attention mechanism to aggregate features instead of pooling 

        self.network_activation = self.hparams.get('activation', 'relu')
        assert self.network_activation in ['relu', 'swish']
        if self.network_activation == 'relu':
            self.network_activation_function = nn.ReLU(True)
        elif self.network_activation == 'swish':
            self.network_activation_function = Swish()

        self.include_global_feature = self.hparams.get('include_global_feature', False)
        # whether to use the global feature from the input point cloud to guide the diffusion model

        self.global_feature_dim = None
        remove_last_activation = self.hparams.get('global_feature_remove_last_activation', True)
        if self.include_global_feature:
            if not self.hparams['pnet_global_feature_architecture'][0][0] == in_fea_dim:
                self.hparams['pnet_global_feature_architecture'][0][0] = in_fea_dim
                print('Have corrected the input dim in global pnet to', in_fea_dim, flush=True)
            if self.use_position_encoding:
                self.hparams['pnet_global_feature_architecture'][0][0] = (
                    self.hparams['pnet_global_feature_architecture'][0][0]+pos_encode_out_dim)
            self.global_feature_dim = self.hparams['pnet_global_feature_architecture'][1][-1]
            self.global_pnet = Pnet2Stage(self.hparams['pnet_global_feature_architecture'][0],
                                            self.hparams['pnet_global_feature_architecture'][1],
                                            bn=self.bn, remove_last_activation=remove_last_activation)

        # time step embedding setting
        # we should include t in the conditional generation network
        # not include t in th refinement network
        include_t = self.hparams['include_t']
        t_dim = self.hparams['t_dim']
        # timestep embedding fc layers
        self.fc_t1 = nn.Linear(t_dim, 4*t_dim)
        self.fc_t2 = nn.Linear(4*t_dim, 4*t_dim)
        self.activation = swish # activation function for t embedding

        # build SA module for the noisy point cloud x_t
        arch = self.hparams['architecture']
        npoint = arch['npoint']#[1024, 256, 64, 16]
        radius = arch['radius']#[0.1, 0.2, 0.4, 0.8]
        nsample = arch['nsample']#[32, 32, 32, 32]
        feature_dim = arch['feature_dim']#[32, 64, 128, 256, 512]
        mlp_depth = arch['mlp_depth']#3
        # if first conv, first conv in_fea_dim + encoder_feature_map_dim[0] -> feature_dim[0]
        # if not first conv, mlp[0] = in_fea_dim + encoder_feature_map_dim[0]
        additional_fea_dim = None
        self.SA_modules = self.build_SA_model(npoint, radius, 
                                nsample, feature_dim, mlp_depth, 
                                in_fea_dim,
                                self.hparams['include_t'], self.hparams["include_class_condition"], 
                                include_global_feature=self.include_global_feature, global_feature_dim=self.global_feature_dim,
                                additional_fea_dim = additional_fea_dim,
                                neighbor_def=arch['neighbor_definition'], activation=self.network_activation,
                                bn=self.bn, attention_setting=self.attention_setting,
                                global_attention_setting=self.global_attention_setting)
        
        # set point upsampling factor
        # this is used in the refinement network, we refine and upsample the input coarse point cloud at the same time
        # point_upsample_factor = self.hparams.get('point_upsample_factor', 1)
        # if point_upsample_factor > 1:
        #     if self.hparams['first_refine_coarse_points']:
        #         point_upsample_factor = point_upsample_factor + 1
        #         if self.hparams['include_displacement_center_to_final_output']:
        #             point_upsample_factor = point_upsample_factor - 1
        #     else:
        #         assert not self.hparams['include_displacement_center_to_final_output']
        #     self.hparams['out_dim'] = int(self.hparams['out_dim'] * (point_upsample_factor))
        
        self.transform_output = self.hparams.get('transform_output', False)
        if self.transform_output:
            input_dim = feature_dim[-1]
            self.fc_lyaer = nn.Sequential(
                nn.Conv1d(input_dim, self.hparams['out_dim'], kernel_size=1),
            )
        

    def forward(self, pointcloud, ts=None, label=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            condition: (B,M,3 + input_channels) tensor, a condition point cloud.
        """

        with torch.no_grad():
            if self.use_position_encoding:
                xyz_ori = pointcloud[:,:,0:3] / self.scale_factor
                xyz_pos_encode = self.pos_encode(xyz_ori)
                pointcloud = torch.cat([pointcloud, xyz_pos_encode], dim=2)

            if self.attach_position_to_input_feature:
                xyz_ori = pointcloud[:,:,0:3] / self.scale_factor
                pointcloud = torch.cat([pointcloud, xyz_ori], dim=2)
                # in this case, the input pointcloud is of shape (B,N,C)
                # the output pointcloud is of shape (B,N,C+3)
                # we want the X not only as position, but also as input feature
                in_fea_dim = self.in_fea_dim - 3
            else:
                in_fea_dim = self.in_fea_dim
            
            xyz, features = self._break_up_pc(pointcloud)
            xyz = xyz / self.scale_factor
            # if pointcloud is of shape BN3, then xyz=pointcloud, features=None
            # if pointcloud is of shape BN(3+C), then xyz is of shape BN3, features is of shape (B,C,N)

        if (not ts is None) and self.hparams['include_t']:
            t_emb = calc_t_emb(ts, self.hparams['t_dim'])
            t_emb = self.fc_t1(t_emb)
            t_emb = self.activation(t_emb)
            t_emb = self.fc_t2(t_emb)
            t_emb = self.activation(t_emb)
        else:
            t_emb = None

        if (not label is None) and self.hparams['include_class_condition']:
            # label should be 1D tensor of integers of shape (B)
            class_emb = self.class_emb(label) # shape (B, condition_emb_dim)
        else:
            class_emb = None
        
        if self.include_global_feature:
            if in_fea_dim > 0:
                input_fea = pointcloud[:,:,3:(3+in_fea_dim)]
                global_input = torch.cat([xyz, input_fea], dim=2)
            else:
                global_input = xyz
            
            global_feature = self.global_pnet(global_input.transpose(1,2))
            
        
        if self.include_global_feature:
            condition_emb = global_feature
            second_condition_emb = class_emb if self.hparams['include_class_condition'] else None
        else:
            condition_emb = class_emb if self.hparams['include_class_condition'] else None
            second_condition_emb = None

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            input_feature = l_features[i]
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], input_feature, t_emb=t_emb, 
                                    condition_emb=condition_emb, second_condition_emb=second_condition_emb,
                                    subset=True, record_neighbor_stats=self.record_neighbor_stats,
                                    pooling=self.pooling)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        out_feature = l_features[-1]
        if self.transform_output:
            out = self.fc_lyaer(out_feature)
        out = torch.transpose(out_feature, 1,2)

        return out, l_xyz, l_features


