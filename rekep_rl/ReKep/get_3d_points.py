import glob
import os
from tqdm import tqdm
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import json

# datetime object containing current date and time
now = datetime.now()
def generate_points_from_depth(depth, proj):
    '''
    :param depth: (B, 1, H, W)
    :param proj: (B, 4, 4)
    :return: point_cloud (B, 3, H, W)
    '''
    batch, height, width = depth.shape[0], depth.shape[2], depth.shape[3]
    inv_proj = torch.inverse(proj)

    rot = inv_proj[:, :3, :3]  # [B,3,3]
    trans = inv_proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    rot_depth_xyz = rot_xyz * depth.view(batch, 1, -1)
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1)  # [B, 3, H*W]
    proj_xyz = proj_xyz.view(batch, 3, height, width)

    return proj_xyz

def write_ply(file, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    o3d.io.write_point_cloud(file, pcd, write_ascii=False)

def get_proj_points_side_cam(cam_K, cam2base, depth_np,):
   # print("Cam K: ", cam_K)
   # print("cam to base: ", cam2base)
   # print("Ensure the depth map is at the initial resolution in order to be compatible with the camera intrinsics,"
   #       " which should be 720x1280.")

    world2cam = cam_K @ np.linalg.inv(cam2base)

    proj_mat_t = torch.from_numpy(world2cam).unsqueeze(0).float()
    depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()

    points = generate_points_from_depth(depth_t, proj_mat_t)
    points = points.detach().numpy()[0].transpose(1, 2, 0) # (h, w, 3)

    return points

def get_adj_Ts():
    adj_transforms = [
        np.array([[1, 0, 0, 0.04],
                  [0, 1, 0, -0.02],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 0],
                  [0, 1, -0.02, 0],
                  [0, 0.02, 1, 0],
                  [0, 0, 0, 1]]),
        # np.array([[1, 0, 0, 0.0],
        #           [0, 1, 0, -0.01],
        #           [0, 0, 1, 0],
        #           [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 0.0],
                  [0, 1, 0, 0.01],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]),
    ]
    T = np.eye(4)
    for i in range(len(adj_transforms)):
        T = adj_transforms[i] @ T

    return T
def get_world_coords(path):
    path_to_depth = os.path.join(path, 'depth_000000000.jpg.npy')
    dep_np=np.load(path_to_depth)

    dep_np = dep_np.astype('int16')
    cam2base = np.array([[0.00966241, 0.75430732, -0.65645039, 0.89545973],
                         [0.9995105, -0.02682037, -0.0161065, 0.03701747],
                         [-0.0297555, -0.65597343, -0.75419724, 0.48401451],
                         [0., 0., 0., 1.]])
    cam2base = get_adj_Ts() @ cam2base

    K_3x3 = np.array([[608.017, 0, 640.24],
                      [0, 607.796, 364],
                      [0, 0, 1]])
    cam_K = np.eye(4)
    cam_K[:3, :3] = K_3x3

    scene_pts = get_proj_points_side_cam(cam_K=cam_K,
                                         cam2base=cam2base,
                                         depth_np=dep_np)
    scene_pts /= 1000
    return scene_pts


def main(path, save = True):
     
    path_to_depth = os.path.join(path, 'depth_000000000.jpg.npy')
    dep_np=np.load(path_to_depth)

    dep_np = dep_np.astype('int16')
    cam2base = np.array([[0.00966241, 0.75430732, -0.65645039, 0.89545973],
                         [0.9995105, -0.02682037, -0.0161065, 0.03701747],
                         [-0.0297555, -0.65597343, -0.75419724, 0.48401451],
                         [0., 0., 0., 1.]])
    cam2base = get_adj_Ts() @ cam2base

    K_3x3 = np.array([[608.017, 0, 640.24],
                      [0, 607.796, 364],
                      [0, 0, 1]])
    cam_K = np.eye(4)
    cam_K[:3, :3] = K_3x3

    scene_pts = get_proj_points_side_cam(cam_K=cam_K,
                                         cam2base=cam2base,
                                         depth_np=dep_np)

    print(scene_pts.shape)
    # each pixel coordinate correspond to a 3d coordinate in world space
    
    scene_pts = np.reshape(scene_pts, (-1, 3)) 

    # convert from mm to m
    scene_pts /= 1000
    
    colors = np.zeros(scene_pts.shape)
    color_pts = np.concatenate([scene_pts, colors], axis=1)

    write_path = os.path.join(path, 'world_coords.ply')
    write_ply(write_path, points=color_pts) 



if __name__ == '__main__':

    path_to_folder = "/nethome/atian31/flash8/repos/ReKep/manual_data/vlm2reward/"

    subfolders = [ f.path for f in os.scandir(path_to_folder) if f.is_dir() ]
   
    for folder in tqdm(subfolders):
       #  for each folder generate 2d -> world state mapping
        main(folder)
 
