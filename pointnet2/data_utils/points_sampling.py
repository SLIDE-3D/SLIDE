from random import randint
from typing import List, Optional, Tuple, Union

import torch
# from pytorch3d import _C

from pytorch3d.ops.utils import masked_gather
import pytorch3d
import numpy as np

import pdb

def sample_farthest_points_naive(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
    initial_points: Optional[torch.Tensor] = None,
    initial_points_lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same Args/Returns as sample_farthest_points
    if initial_points is not None, we keep the first several points to inital points
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)

    if lengths.shape[0] != N:
        raise ValueError("points and lengths must have same batch dimension.")

    if not initial_points is None:
        # initial_points is of shape N, P1, D
        N2, P2, D2 = initial_points.shape
        assert N == N2 and D==D2
        if initial_points_lengths is None:
            initial_points_lengths = torch.full((N,), P2, dtype=torch.int64, device=device)
        assert (initial_points_lengths > lengths).sum() == 0
        assert (initial_points_lengths > K).sum() == 0
        assert not random_start_point
        new_points = torch.zeros(N, P+P2, D, dtype=points.dtype, device=device)
        new_lengths = lengths + initial_points_lengths
        for i in range(N):
            new_points[i, 0:new_lengths[i], :] = torch.cat([initial_points[i, 0:initial_points_lengths[i], :], points[i, 0:lengths[i], :]])
        
        points = new_points
        lengths = new_lengths

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Find max value of K
    max_K = torch.max(K)

    # List of selected indices from each batch element
    all_sampled_indices = []

    for n in range(N):
        # Initialize an array for the sampled indices, shape: (max_K,)
        sample_idx_batch = torch.full(
            (max_K,), fill_value=-1, dtype=torch.int64, device=device
        )

        # Initialize closest distances to inf, shape: (P,)
        # This will be updated at each iteration to track the closest distance of the
        # remaining points to any of the selected points
        closest_dists = points.new_full(
            (lengths[n],), float("inf"), dtype=torch.float32
        )

        # Select a random point index and save it as the starting point
        selected_idx = randint(0, lengths[n] - 1) if random_start_point else 0
        sample_idx_batch[0] = selected_idx

        # If the pointcloud has fewer than K points then only iterate over the min
        k_n = min(lengths[n], K[n])

        # Iteratively select points for a maximum of k_n
        for i in range(1, k_n):
            # Find the distance between the last selected point
            # and all the other points. If a point has already been selected
            # it's distance will be 0.0 so it will not be selected again as the max.
            dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
            dist_to_last_selected = (dist ** 2).sum(-1)  # (P - i)

            # If closer than currently saved distance to one of the selected
            # points, then updated closest_dists
            closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

            # The aim is to pick the point that has the largest
            # nearest neighbour distance to any of the already selected points
            if initial_points is not None and i < initial_points_lengths[n]:
                # force to select the i-th points
                selected_idx = i
            else:
                selected_idx = torch.argmax(closest_dists)
                
            sample_idx_batch[i] = selected_idx

        # Add the list of points for this batch to the final list
        all_sampled_indices.append(sample_idx_batch)

    all_sampled_indices = torch.stack(all_sampled_indices, dim=0)

    # Gather the points
    all_sampled_points = masked_gather(points, all_sampled_indices)

    # Return (N, max_K, D) subsampled points and indices
    return all_sampled_points, all_sampled_indices

def append_points_to_keypoints(points, initial_points, K, lengths=None, initial_points_lengths=None, device=torch.device('cpu'),
                            only_return_appended_points=False):
    # we make the keypoints have at least K points
    return_array = isinstance(points, np.ndarray)
    if return_array:
        points = torch.from_numpy(points).to(device)
        initial_points = torch.from_numpy(initial_points).to(device)
    no_batch = len(points.shape) == 2
    if no_batch:
        points = points.unsqueeze(0)
        initial_points = initial_points.unsqueeze(0)

    # pdb.set_trace()
    if initial_points.shape[1] >= K:
        assert lengths is None and initial_points_lengths is None
        sampled_points = initial_points
        initial_K = initial_points.shape[1]
        sampled_indices = torch.ones(sampled_points.shape[0], initial_K, dtype=torch.int64, device=sampled_points.device) * (-1)
    else:
        sampled_points, sampled_indices = sample_farthest_points_naive(points, lengths=lengths, K=K,
                random_start_point=False, initial_points=initial_points, initial_points_lengths=initial_points_lengths)
    
    if only_return_appended_points:
        assert lengths is None and initial_points_lengths is None
        initial_K = initial_points.shape[1]
        sampled_points = sampled_points[:, initial_K:, :]
        sampled_indices = sampled_indices[:, initial_K:]

    if no_batch:
        sampled_points = sampled_points[0]
        sampled_indices = sampled_indices[0]
    if return_array:
        sampled_points = sampled_points.detach().cpu().numpy()
        sampled_indices = sampled_indices.detach().cpu().numpy()
    return sampled_points, sampled_indices

def sample_keypoints(x, K, add_centroid=True, device=torch.device('cpu'), random_subsample=False):
    # x is a tensor or numpy array of shape N,P,D or P,D
    return_array = isinstance(x, np.ndarray)
    if return_array:
        x = torch.from_numpy(x).to(device)

    no_batch = len(x.shape) == 2
    if no_batch:
        x = x.unsqueeze(0)

    if add_centroid:
        centroid = x.mean(dim=1, keepdim=True) # N,1,D
        x = torch.cat([centroid, x], dim=1) # N,P+1,D

    if random_subsample:
        assert not add_centroid
        idx = torch.randperm(x.shape[1])
        idx = idx[0:K]
        selected_points = x[:,idx,:]
        selected_idx = idx.unsqueeze(0) # of shape 1,K
    else:
        selected_points, selected_idx = pytorch3d.ops.sample_farthest_points(x, K=K, random_start_point=not add_centroid)
    # selected_points is of shape (N, K, D), selected_idx is of shape (K, D)

    if no_batch:
        selected_points = selected_points[0]
        selected_idx = selected_idx[0]
    if return_array:
        selected_points = selected_points.detach().cpu().numpy()
        selected_idx = selected_idx.detach().cpu().numpy()
    
    return selected_points, selected_idx



