import torch
import torch.nn as nn
from models.point_upsample_decoder import PointUpsampleDecoder

import pdb

class KeypointDecoder(nn.Module):

    def __init__(self, config_list, feature_dim):
        super().__init__()
        self.decoders = nn.ModuleList()
        for i in range(len(config_list)):
            # pdb.set_trace()
            self.decoders.append(PointUpsampleDecoder(config_list[i], in_dim=feature_dim))
            if 'decoder_feature_dim' in config_list[i]['architecture'].keys():
                feature_dim = config_list[i]['architecture']['decoder_feature_dim'][0] + config_list[i]['feature_mapper_setting']['out_dim']
            else:
                feature_dim = config_list[i]['architecture']['feature_dim'][-1] + config_list[i]['feature_mapper_setting']['out_dim']
                # if config_list[i].get('transform_output', False):
                #     feature_dim = config_list[i]['architecture']['feature_dim'][-1] + config_list[i]['feature_mapper_setting']['out_dim']
                # else:
                #     feature_dim = config_list[i]['out_dim'] + config_list[i]['feature_mapper_setting']['out_dim']
        

    def forward(self, xyz0, features0, xyz1, ts=None, label=None):
        # xyz0 is of shape B,N1,3, features0 is of shape B,N1,C1, which are features at points xyz0
        # xyz1 are user provided key points of shape B,N2,3 or B,N2,C2 if you want xyz1 to carry additional features

        l_xyzs = [xyz0, xyz1]
        l_features = [features0]
        for i, decoder in enumerate(self.decoders):
            new_feature, new_xyz = decoder(l_xyzs[i][:,:,0:3], l_features[i], l_xyzs[i+1], ts=ts, label=label)
            l_xyzs.append(new_xyz)
            l_features.append(new_feature)
         
        return l_xyzs


