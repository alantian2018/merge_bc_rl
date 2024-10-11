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
from ReKep.utils_rl import *
from tqdm import tqdm
# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


def playback_trajectory_with_env(
    env, 
    initial_state, 
    states, 
    actions=None, 
    render=False, 
    video_writer=None, 
    video_skip=1, 
    camera_height=1024,
    camera_width = 1024,
    camera_names=None,
    first=False,
):

    assert isinstance(env, EnvBase)
    DEPTH_HIST =[]
    PIXEL_HIST = []
    EE_HIST = []
    write_video = (video_writer is not None)
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    traj_len = states.shape[0]
    action_playback = (actions is not None)
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    for i in tqdm(range(traj_len)):

        if i == 0:
            keypoints_path = os.path.join(args.task_dir, 'annotations','first_frame_annotated.json')
            with open (keypoints_path) as f:
                keypoints=json.load(f)
            objects = get_objects_from_keypoints(keypoints)

            cam = camera_names[0]
            LOCAL_POSE = get_local_pose(env,keypoints,
                                        objects, 
                                        camheight = camera_height,
                                        camwidth = camera_width,
                                        camera = cam)
            
            
        
        ee, NEW_DEPTHS, NEW_PIXELS = track_keypoints(env,
                                                     LOCAL_POSE,
                                                     objects, 
                                                     camera=cam,
                                                     camheight=camera_height,
                                                     camwidth=camera_width
                                                    )
    
        DEPTH_HIST.append(NEW_DEPTHS)
        PIXEL_HIST.append(NEW_PIXELS)
        EE_HIST.append(ee)
        
    
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=camera_height, width=camera_width, camera_name=cam_name))
                
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally    
                
                video_img = color_keypoints(NEW_PIXELS, video_img)
                video_writer.append_data(video_img)

            video_count += 1

        if action_playback:
            env.step(actions[i])
            if i < traj_len - 1:
                state_playback = env.get_state()["states"]
            

        if first:
            break
   
    with open (os.path.join(args.task_dir,'ground_truths', 'keypoints.pkl'),'wb') as f:
        pickle.dump( {'ee': EE_HIST, 'depth' : DEPTH_HIST, 'pixel' : PIXEL_HIST}, f)


def playback_dataset(args):
 
    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]
 
 
    # create environment only if not playing back with observations
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    
    
    env = EnvUtils.create_env_for_data_processing(env_meta=env_meta,
                                                        camera_names=args.render_image_names, 
                                                        camera_height=1024, 
                                                        camera_width=1024, 
                                                        reward_shaping=False,
                                                        use_depth_obs=True, 
                                                        render=False,
                                                        render_offscreen=True)

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
  
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    demos = demos[:1]
 
    video_writer = imageio.get_writer(os.path.join(args.task_dir, 'ground_truths/rollout_rekep_constraints.mp4'), fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
       
        actions = f["data/{}/actions".format(ep)][()]

        playback_trajectory_with_env(
            env=env, 
            initial_state=initial_state, 
            states=states, actions=actions, 
            render=False, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
        )

    f.close()
    
    video_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )

    
    
    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--task_dir",
        type=str,
        required=True,
        help="Folder to task, eg. ./data/can or ./data/cube ",
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

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(args.task_dir, 'ground_truths'), exist_ok=True)
    playback_dataset(args)