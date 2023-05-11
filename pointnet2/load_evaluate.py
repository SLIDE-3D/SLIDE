import imp
import numpy as np 
from pytorch3d.loss.chamfer import chamfer_distance
import torch, argparse
import torch.nn as nn
import ipdb

def fscore(dist1, dist2, threshold=0.0001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

def calc_cd(gen_samples, ref_samples, calc_f1=False, f1_threshold=0.0001):

    dist1, dist2, _ = chamfer_distance(gen_samples, ref_samples, batch_reduction=None, point_reduction=None)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2, threshold=f1_threshold)
        # f1 = torch.zeros(output.shape[0], device=output.device)
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t


class Chamfer_F1(nn.Module):
    def __init__(self, f1_threshold=0.0001):
        super().__init__()
        self.f1_threshold = f1_threshold

    def forward(self, xyz1, xyz2):
        # xyz1 and xyz2 are of shape B,N,3
        # cd_p, cd_t, f1 are of shape N
        cd_p, cd_t, f1 = calc_cd(xyz1, xyz2, calc_f1=True, f1_threshold=self.f1_threshold)
        return cd_p, cd_t, f1

def normalize_point_cloud(all_points, normalize_std_per_axis = True, normalize_per_shape = True, all_points_mean = None, 
                            all_points_std = None,input_dim = 3, box_per_shape = False):
    """
    Normalize a point cloud to have zero mean and unit variance.
    """
    if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
        all_points_mean = all_points_mean
        all_points_std = all_points_std
    elif normalize_per_shape:  # per shape normalization
        B, N = all_points.shape[:2]
        all_points_mean = all_points.mean(axis=1).reshape(B, 1, input_dim)
        if normalize_std_per_axis:
            all_points_std = all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
        else:
            all_points_std = all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
    elif box_per_shape:
        B, N = all_points.shape[:2]
        all_points_mean = all_points.min(axis=1).reshape(B, 1, input_dim)

        all_points_std = all_points.max(axis=1).reshape(B, 1, input_dim) - all_points.min(axis=1).reshape(B, 1, input_dim)

    else:  # normalize across the dataset
        all_points_mean = all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
        if normalize_std_per_axis:
            all_points_std = all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
        else:
            all_points_std = all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

    all_points = (all_points - all_points_mean) / all_points_std

    return all_points

def normalize_per_shape(all_points):
    """
    move the points cloud to centroid, choose the longest axis length to divide.
    """
    B, N = all_points.shape[:2]
    ipdb.set_trace()
    all_points_mean = all_points.mean(axis=1).reshape(B, 1, 3)
    all_points_std = all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, 3)
    all_points = (all_points - all_points_mean) / all_points_std

    return all_points



if __name__ == "__main__":
    """
    from the directory of dir1 and dir2, load the *.npz file as dict data type. 
    Then evaluate the CD and EMD of two sets of points cloud.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir1', type=str, default='data/pointflow_1.npz')
    parser.add_argument('--dir2', type=str, default='data/pointflow_2.npz')
    parser.add_argument('--threshold', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--normalize_std_per_axis', type=bool, default=True)
    parser.add_argument('--normalize_per_shape', type=bool, default=True)
    args = parser.parse_args()

    dir1 = args.dir1
    dir2 = args.dir2

    # load the data
    data1 = np.load(dir1)
    data2 = np.load(dir2)

    # get the keys, keys are 'points'
    key = 'points'
    p1 = data1[key]
    p2 = data2[key]
    if args.normalize:
        p1 = normalize_point_cloud(p1, args.normalize_std_per_axis, args.normalize_per_shape)
        p2 = normalize_point_cloud(p2, args.normalize_std_per_axis, args.normalize_per_shape)

    # calculate the CD 
    chamfer_f1 = Chamfer_F1()
    cd_p, cd_t, f1 = chamfer_f1(p1, p2)
    print('cd_p: ', cd_p)
    print('cd_t: ', cd_t)
    print('f1: ', f1)
    # print('EMD: ', emd)
