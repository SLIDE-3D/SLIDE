import torch
import copy
from pointnet2_ops import pointnet2_utils
import numpy as np
import os
import pdb

def get_bounding_box(points):
    # points is a numpy array of shape B,N,3 or N,3
    if len(points.shape) == 3:
        reduction_dim = 1
    elif len(points.shape) == 2:
        reduction_dim = 0
    else:
        raise Exception('points is of shape', points)

    minn = points.min(axis=reduction_dim)
    maxx = points.max(axis=reduction_dim)
    center = (maxx + minn) / 2
    length = maxx - minn
    centroid = points.mean(axis=reduction_dim)
    result = {'max':maxx, 'min':minn, 'center':center, 'length':length, 'centroid':centroid}

    return result