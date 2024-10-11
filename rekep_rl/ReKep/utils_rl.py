import numpy as np
import open3d as o3d
import torch
import glob
import os
from tqdm import tqdm
from datetime import datetime
from PIL import Image
import json
from robomimic.utils.env_utils import create_env_from_metadata
from robomimic.utils.file_utils import get_env_metadata_from_dataset
import robosuite.utils.camera_utils as CameraUtils
import robosuite.utils.transform_utils as TUtils
from robomimic.envs.env_robosuite import EnvRobosuite

########################################
# REKEP environment manipulation utils #
########################################
def get_xpos_xquat(env, obj):
    if isinstance(env, EnvRobosuite):
        # robomimic wrapper
        xpos = env.base_env.sim.data.get_body_xpos(obj) 
        xquat = env.base_env.sim.data.get_body_xquat(obj)
    else:
        xpos = env.sim.data.get_body_xpos(obj)
        xquat = env.sim.data.get_body_xquat(obj)

    w, x, y, z = xquat
    xquat =  np.array([ x, y, z, w])
    return xpos, xquat

def get_objects_from_keypoints(keypoints):
    objs = []
    for i in keypoints:
        objs.append((i, keypoints[i]['no_keypoints'],))

    return objs

def get_local_pose(env, keypoints, objs, camera, camwidth=1024, camheight=1024):

    initial_depths = env.get_observation()[f'{camera}_depth']
     
    intrinsics = env.get_camera_intrinsic_matrix(camera, camwidth, camheight)
    extrinsics = env.get_camera_extrinsic_matrix(camera)

    LOCAL_POSE = dict()
  
    for obj, number in objs:

        # get keypoint pixels
        keypoints_obj = keypoints[obj]['keypoints']

        # transform to 3d (x,y,z) position
        keypoints_d = get_depth_from_keypts(initial_depths, keypoints_obj, intrinsics, extrinsics, camwidth, camheight)

        keypoints_homogeneous = np.expand_dims(xyz_to_homogenoeous(keypoints_d), axis= -1)
        
        xpos,xquat = get_xpos_xquat(env, obj)

        original_pose_obj = np.expand_dims(TUtils.pose2mat ((xpos,xquat,)), axis = 0)

        # get pose of keypoints w.r.t the object
        T_inv = np.linalg.inv(original_pose_obj)
        LOCAL_POSE_obj = T_inv @ keypoints_homogeneous

        
        LOCAL_POSE[obj] = LOCAL_POSE_obj

    return LOCAL_POSE 

def track_keypoints(env, LOCAL_POSE, objs,  camera, camheight=1024, camwidth=1024):
    
    # NEW_DEPTHS = T_new @ (T_old)^-1 @ init_keypoints
    #                       --------LOCAL_POSE--------
    #                               no x 4 x 1 
     
    depths = []
    pixels = []
    for obj in objs:
        obj = obj[0]

        # get new pose of object
        xpos, xquat = get_xpos_xquat(env, obj)
        new_pose =  np.expand_dims(TUtils.pose2mat((xpos,xquat,)), axis=0)
        
        object_pose = LOCAL_POSE[obj]

        NEW_DEPTHS = new_pose @ object_pose
        
        NEW_DEPTHS = np.reshape(NEW_DEPTHS, (-1,4))
        
        #drop last column
        NEW_DEPTHS = NEW_DEPTHS[:,:-1]
         
        NEW_PIXELS = world2pixel(env, NEW_DEPTHS, camera, camheight, camwidth)

        depths.append(NEW_DEPTHS)
        pixels.append(NEW_PIXELS)
    
    depths = np.vstack(depths)
    pixels = np.vstack(pixels)
      
    
    ee = env.get_observation()['robot0_eef_pos']
    return ee, depths, pixels
    

def color_keypoints(keypoints, image):
    radius = 2
    for i in keypoints:
        x_center,y_center = i
        x_low = max(x_center - radius, 0)
        x_high = min(x_center + radius, image.shape[0])
        y_low = max(y_center - radius, 0)
        y_high = min(y_center + radius, image.shape[1])
        image[x_low:x_high, y_low:y_high] = [57, 255, 20] 

    return image


def xyz_to_homogenoeous(keypoints):
    filler = np.ones((keypoints.shape[0],1,))
    return np.concatenate((keypoints,filler), axis= 1)


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

def get_proj_points_side_cam(cam_K, cam2base, depth_np,):
  
    world2cam = cam_K @ np.linalg.inv(cam2base)

    proj_mat_t = torch.from_numpy(world2cam).unsqueeze(0).float()
    depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()

    points = generate_points_from_depth(depth_t, proj_mat_t)
    points = points.detach().numpy()[0].transpose(1, 2, 0) # (h, w, 3)

    return points

def write_ply(file, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    o3d.io.write_point_cloud(file, pcd, write_ascii=False)

def get_depth_map(initial_depths, intrinsics, extrinsics,camwidth, camheight):

    K_3x3 = intrinsics
    cam_K = np.eye(4)
    cam_K[:3, :3] = K_3x3
    intial_depths = np.squeeze(initial_depths)
 
    out = get_proj_points_side_cam(cam_K,extrinsics,initial_depths)

    out = np.reshape(out,(-1,3))
    
    colors = np.zeros(out.shape)
    color_pts = np.concatenate([out, colors], axis=1)

    #write_ply('debug.ply',color_pts)
    point_cloud = np.reshape(out, (camheight,camwidth,3))
   
    return point_cloud 

def world2pixel(env, pts, camera, camheight=1024, camwidth=1024):
    world2pixel = env.get_camera_transform_matrix(camera, camheight, camwidth)
    pixels = CameraUtils.project_points_from_world_to_camera(pts, world2pixel, camera_height=camheight, camera_width=camwidth) 
    return pixels

def get_depth_from_keypts(initial_depths, keypoints, intrinsics, extrinsics, camwidth, camheight):
    point_cloud = get_depth_map(initial_depths, intrinsics, extrinsics, camwidth, camheight)
   
    transformed_points = []

    for i in (keypoints):
        transformed_points.append(point_cloud[i[0]][i[1]])
    return np.asarray(transformed_points)
    
