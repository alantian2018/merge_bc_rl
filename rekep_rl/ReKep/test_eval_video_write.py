import os
import json
import h5py
import argparse
import imageio
import numpy as np
import pickle
import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.utils.vis_utils import depth_to_rgb
from robomimic.envs.env_base import EnvBase, EnvType
from robosuite.utils.transform_utils import pose2mat
from sims.extract_keypoints import *
from train_rl import *
import matplotlib.pyplot as plt
# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


def track_keypoints(env , name , LOCAL_POSE, o):
    xpos = env.base_env.sim.data.get_body_xpos(name)
    xquat = env.base_env.sim.data.get_body_xquat(name)
    new_pose =  pose2mat((xpos,xquat,))
 
    
    NEW_DEPTHS = np.hstack((new_pose @ LOCAL_POSE[:,:2], o @ LOCAL_POSE[:,2:],))
    
    NEW_DEPTHS = np.transpose(NEW_DEPTHS)
    
    #drop last column
    NEW_DEPTHS = NEW_DEPTHS[:,:-1]
   
    NEW_PIXELS = world2pixel(env,NEW_DEPTHS)

    ee = env.get_observation()['robot0_eef_pos']
    return ee, NEW_DEPTHS, NEW_PIXELS
    

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

def playback_trajectory_with_env(
    env, 
    initial_state, 
    states, 
    actions=None, 
    render=False, 
    video_writer=None, 
    video_skip=1, 
    camera_names=None,
    first=False,
):

   # assert isinstance(env, EnvBase)
    DEPTH_HIST =[]
    PIXEL_HIST = []
    EE_HIST = []
    write_video = (video_writer is not None)
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    ## get depths of first frame here....
     
    initial_depths =env.get_observation()['agentview_depth']
    traj_len = states.shape[0]
    action_playback = (actions is not None)
    if action_playback:
        assert states.shape[0] == actions.shape[0]
    bruh = []
    for i in range(traj_len):
        if i == 0:
            keypoints_path= '/nethome/atian31/flash8/repos/ReKep/manual_data/sims/can/annotated.json'
            with open (keypoints_path) as f:
                keypts=json.load(f)
            keypoints  = keypts['keypoints']
    
            intrinsics = env.get_camera_intrinsic_matrix('agentview')
            extrinsics = env.get_camera_extrinsic_matrix('agentview')
            keypoints_d = get_depth_from_keypts(initial_depths, keypoints, intrinsics, extrinsics)
             
            keypoints_homogeneous = xyz_to_homogenoeous(keypoints_d)
            xpos = env.base_env.sim.data.get_body_xpos(args.object) 
            xquat = env.base_env.sim.data.get_body_xquat(args.object)

           
            o = pose2mat ((xpos,xquat,))
            LOCAL_POSE = np.linalg.inv(o) @ np.transpose(keypoints_homogeneous)
            
        
        ee, NEW_DEPTHS, NEW_PIXELS = track_keypoints(env, args.object, LOCAL_POSE,o)

        DEPTH_HIST.append(NEW_DEPTHS)
        PIXEL_HIST.append(NEW_PIXELS)
        EE_HIST.append(ee)
        print(f'iter {i}: {NEW_PIXELS}')

        
        if action_playback:
            env.step(actions[i])
            if i < traj_len - 1:
                state_playback = env.get_state()["states"]
        else:
            env.reset_to({"states" : states[i]})

        # video render
    
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally    
                
                video_img = color_keypoints(NEW_PIXELS, video_img)
                video_writer.append_data(video_img)

            video_count += 1

        if first:
            break
    assert (len(EE_HIST) == len(DEPTH_HIST) == len(PIXEL_HIST))
    with open (os.path.join(os.path.dirname(args.video_path), 'keypoints.pkl'),'wb') as f:
        pickle.dump( {'ee': EE_HIST, 'depth' : DEPTH_HIST, 'pixel' : PIXEL_HIST}, f)


def playback_dataset(args):
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert not args.use_actions, "playback with observations is offline and does not support action playback"

    if args.render_depth_names is not None:
        assert args.use_obs, "depth observations can only be visualized from observations currently"

    # create environment only if not playing back with observations
    if not args.use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        """dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[ ],
                    depth=['agentview_depth']
                ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)"""

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
       
        env = EnvUtils.create_env_for_data_processing(env_meta=env_meta,
                                                            camera_names=args.render_image_names, 
                                                            camera_height=512, 
                                                            camera_width=512, 
                                                            reward_shaping=False,
                                                            use_depth_obs=True, 
                                                            render= args.render,
                                                            render_offscreen=write_video)

        # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)
        env = RoboMimic_RL_Env(
        path_to_dataset = '/nethome/atian31/flash8/repos/robomimic/datasets/can/ph/low_dim_v141.hdf5',
        path_to_keypoints = '/nethome/atian31/flash8/repos/ReKep/manual_data/sims/can/annotated.json',
        path_to_VLM_query = '/nethome/atian31/flash8/repos/ReKep/vlm_query/2024-09-13_02-23-45_have_the_robot_lift_the_can_and_place_it_in_the_bin_with_the_appropriate_outline',
        max_episode_steps = 1000,
        evaluate = True
    )

    

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]

        playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            states=states, actions=actions, 
            render=args.render, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
        )

    f.close()
    if write_video:
        video_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.


    parser.add_argument(
        '--object',
        type=str,
        default='cube_main',
        help='object to track'
    )
    # name of object to be tracked

    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )
    

    

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # depth observations to use for writing to video
    parser.add_argument(
        "--render_depth_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) depth observation(s) to use for rendering to video"
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    args = parser.parse_args()
    playback_dataset(args)