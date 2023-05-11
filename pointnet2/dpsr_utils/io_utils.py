# import open3d as o3d
import os
import torch
import numpy as np
import pytorch3d.io as ptio
from pyntcloud import PyntCloud
import pandas as pd

import pdb

def save_mesh(save_file, vertices, faces, normals=None):
    # vertices, faces, normals are tensors or numpy arrays of shape (N,3) (F,3) (N,3)
    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(np.ascontiguousarray(vertices))
    if isinstance(faces, np.ndarray):
        # pdb.set_trace()
        faces = torch.from_numpy(np.ascontiguousarray(faces))
    if isinstance(normals, np.ndarray):
        normals = torch.from_numpy(np.ascontiguousarray(normals))
    ptio.save_ply(save_file, vertices, faces=faces, verts_normals=normals)
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh.triangles = o3d.utility.Vector3iVector(faces)
    # if not normals is None:
    #     mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    # o3d.io.write_triangle_mesh(save_file, mesh)

def batch_pynt_save_pcd(save_dir, save_prefix, points, batch_info=None, normals=None, indicator=None, start_idx = 0):
    # points, normals are tensors or numpy arrays of shape B,N,3
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(normals, torch.Tensor):
        normals = normals.detach().cpu().numpy()
    if isinstance(indicator, torch.Tensor):
        indicator = indicator.detach().cpu().numpy()
    B = points.shape[0]
    for i in range(B):
        points_i = points[i]
        normals_i = None if normals is None else normals[i]
        indicator_i = None if indicator is None else indicator[i]
        if batch_info is None:
            save_name = save_prefix + '_' + str(start_idx+i).zfill(5) + '.ply'
        else:
            save_name = batch_info[i] + '_' + str(start_idx+i).zfill(5) + '.ply'
        save_name = os.path.join(save_dir, save_name)
        pynt_save_pcd(save_name, points_i, normals=normals_i, indicator=indicator_i)

def pynt_save_pcd(save_file, points, normals=None, indicator=None):
    # points, normals are numpy arrays of shape N,3
    # indicator is a numpy array that contains 1 and -1 of shape N
    data_dict = {'x':points[:,0], 'y':points[:,1], 'z':points[:,2]}
    if not normals is None:
        data_dict.update({'nx':normals[:,0], 'ny':normals[:,1], 'nz':normals[:,2]})
    if not indicator is None:
        original_point = [255,0,0]
        mirrored_point = [0,255,0]
        color = [original_point if idx > 0 else mirrored_point for idx in indicator]
        color = np.array(color).astype(np.uint8)
        data_dict.update({'red':color[:,0], 'green':color[:,1], 'blue':color[:,2]})
    df = pd.DataFrame(data=data_dict)
    cloud = PyntCloud(df)
    cloud.to_file(save_file)

def save_pcd(save_file, points, normals=None):
    # points, normals are tensors of shape N,3
    ptio.save_ply(save_file, points, verts_normals=normals)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # if not normals is None:
    #     pcd.normals = o3d.utility.Vector3dVector(normals)
    # o3d.io.write_point_cloud(save_file, pcd)

def batch_save_pcd(save_dir, save_prefix, points, normals=None, start_idx = 0):
    # points, normals are tensors or numpy arrays of shape B,N,3
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(np.ascontiguousarray(points))
    if isinstance(normals, np.ndarray):
        normals = torch.from_numpy(np.ascontiguousarray(normals))
    B = points.shape[0]
    for i in range(B):
        points_i = points[i]
        normals_i = None if normals is None else normals[i]
        save_name = save_prefix + '_' + str(start_idx+i).zfill(5) + '.ply'
        save_name = os.path.join(save_dir, save_name)
        save_pcd(save_name, points_i, normals=normals_i)
    # if isinstance(points, torch.Tensor):
    #     points = points.detach().cpu().numpy()
    # if isinstance(normals, torch.Tensor):
    #     normals = normals.detach().cpu().numpy()
    # B = points.shape[0]
    # for i in range(B):
    #     points_i = points[i]
    #     normals_i = None if normals is None else normals[i]
    #     save_name = save_prefix + '_' + str(start_idx+i).zfill(5) + '.ply'
    #     save_name = os.path.join(save_dir, save_name)
    #     save_pcd(save_name, points_i, normals=normals_i)
