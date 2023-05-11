import torch
import numpy as np

def point_upsample(coarse, displacement, point_upsample_factor, include_displacement_center_to_final_output=False,
                    output_scale_factor_value=0.001, first_refine_coarse_points=False):
    # coarse is of shape B,N,F, F is the feature dimension
    # displacement is of shape (B,N,F*point_upsample_factor) or (B,N,F*(point_upsample_factor+1))

    # if first_refine_coarse_points and include_displacement_center_to_final_output
    #   point_upsample_factor = point_upsample_factor
    # elif first_refine_coarse_points and not include_displacement_center_to_final_output
    #   point_upsample_factor = point_upsample_factor + 1
    # elif not first_refine_coarse_points and not include_displacement_center_to_final_output
    #   point_upsample_factor = point_upsample_factor
    # else
    #   raise Exception

    if not first_refine_coarse_points:
        assert not include_displacement_center_to_final_output
    
    B,N,F = coarse.size()
    grid_scale_factor = 1 / np.sqrt(point_upsample_factor)
    
    if first_refine_coarse_points:
        grid_displacement = displacement[:,:,F:] * grid_scale_factor
        center_displacement = displacement[:,:,0:F]
        intermediate_refined_X = coarse + center_displacement * output_scale_factor_value
        if include_displacement_center_to_final_output:
            grid_displacement = grid_displacement.view(B, N, point_upsample_factor-1, F)
        else:
            grid_displacement = grid_displacement.view(B, N, point_upsample_factor, F)
    else:
        grid_displacement = displacement * grid_scale_factor
        intermediate_refined_X = coarse
        grid_displacement = grid_displacement.view(B, N, point_upsample_factor, F)

    # include_displacement_center_to_final_output = pointnet_config['include_displacement_center_to_final_output']
    upsampled_X = intermediate_refined_X.unsqueeze(2) + grid_displacement * output_scale_factor_value
    # (B, N, point_upsample_factor-1, F) or (B, N, point_upsample_factor, F)
    upsampled_X = upsampled_X.reshape(B, -1, F)
    if include_displacement_center_to_final_output:
        refined_X = torch.cat([upsampled_X, intermediate_refined_X], dim=1).contiguous()
    else:
        refined_X = upsampled_X.contiguous()
    # refined_X is of shape (B, N*point_upsample_factor, F)
    return refined_X