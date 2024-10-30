"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""
import argparse
import numpy as np
import os
import h5py
import json
import gym
from gym import spaces, Env
from collections import OrderedDict
from rekep_rl.ReKep.utils_rl import *
from rekep_rl.benchmark import get_dense_reward_constraint, get_reward_func
from stable_baselines3.common.env_checker import check_env
import robosuite
from robosuite.wrappers import Wrapper
import robosuite.utils.transform_utils as T
from stable_baselines3 import PPO
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

class CustomRewardGymWrapper(gym.Env,Wrapper ):
    metadata = None
    render_mode = None

    def __init__(self,
                 env,
                 objects,
                 initial_local_pose,
                 reward_functions,
                 camheight=1024, 
                 camwidth=1024,
                 camera='agentview',
                 low_dim = True,
                 keys=[ "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    "object"]):

        # Run super method
        super().__init__(env=env)

        self._is_v1 = (robosuite.__version__.split(".")[0] == "1")
        self.LOCAL_POSE = initial_local_pose
        self.objects = objects
        self.reward_functions = reward_functions
        self.camera = camera
        self.camheight = camheight
        self.camwidth = camwidth
        
      
        #self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.ee = None
        self.pixels = None
        self.keypoints = None
        

        if len(self.reward_functions)==0:
            print('Using default shaped rewards since you did not provide VLM query')
            self.reward_functions=None

        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if not low_dim:
                if self.env.use_camera_obs:
                    keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        self.low_dim = low_dim
      

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        flat_ob = self.reset()
        obs = self.get_observation()
        self.modality_dims = OrderedDict()
        for key in obs.keys():
            self.modality_dims[key] =  obs[key].shape[0]
    #    print(f'mod dim {self.modality_dims}')
        
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        
        self.observation_space = spaces.Box(low, high)
        
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)
       
         

    def custom_reward(self):
        self.ee, self.keypoints, self.pixels = track_keypoints(self, 
                                                               self.LOCAL_POSE, 
                                                               self.objects,
                                                               self.camera,
                                                               self.camheight,
                                                               self.camwidth)

        # compute sparse rewards
        if (self.env._check_success()):
            # return one if we successful
            return 1
        
        staged_rewards = get_dense_reward_constraint(self.ee, self.keypoints, self.reward_functions)
        return staged_rewards

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in obs_dict:
            if verbose:
                print("adding key: {}".format(key))
            obs = obs_dict[key]
            ob_lst.append(np.array(obs).flatten())
        return np.concatenate(ob_lst)

    def reset(self, seed=None, options=None):
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")

        self.env.reset()
        self.steps = 0
        out = self.get_observation()
        

        
        return self._flatten_obs(out)
         

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml
        
        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        self.steps = 0
        should_ret = False
        if "model" in state:
            self.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml
                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = self.env.edit_model_xml(state["model"])
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()
            if not self._is_v1:
                # hide teleop visualization after restoring from model
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            self.env.sim.set_state_from_flattened(state["states"])
            self.env.sim.forward()
            should_ret = True

        if "goal" in state:
            self.set_goal(**state["goal"])
        if should_ret:
            # only return obs if we've done a forward call - otherwise the observations will be garbage
            
            return self.get_flat_observation()
        
        return None

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)

        if self.reward_functions is not None:
            reward = self.custom_reward()

        done = (reward == 1)
        
       # if (self.steps == self.max_episode_steps):
       #     done = True
        self.steps += 1

         
        return self._flatten_obs(self.get_observation()), reward, terminated, info
       

    def render(self):
        return self.env.render(mode="rgb_array", height=self.camheight, width=self.camwidth, camera_name=self.camera) 

    def compute_reward(self, achieved_goal, desired_goal, info):

        if self.reward_functions is None:
            return self.env.reward()
        print('using custom reward')
        return self.custom_reward()

    def get_camera_intrinsic_matrix(self, camera_name, camera_height, camera_width):

        cam_id = self.env.sim.model.camera_name2id(camera_name)
        
        fovy = self.env.sim.model.cam_fovy[cam_id]
     
        f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
        K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
        return K

    def get_camera_extrinsic_matrix(self, camera_name):
         
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        camera_pos = self.env.sim.data.cam_xpos[cam_id]
        camera_rot = self.env.sim.data.cam_xmat[cam_id].reshape(3, 3)
        R = T.make_pose(camera_pos, camera_rot)

        # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        R = R @ camera_axis_correction
        return R

    def get_camera_transform_matrix(self, camera_name, camera_height, camera_width):
        R = self.get_camera_extrinsic_matrix(camera_name=camera_name)
        K = self.get_camera_intrinsic_matrix(
            camera_name=camera_name, camera_height=camera_height, camera_width=camera_width
        )
        K_exp = np.eye(4)
        K_exp[:3, :3] = K

        # Takes a point in world, transforms to camera frame, and then projects onto image plane.
        return K_exp @ T.pose_inv(R)

    def get_observation(self):
        di = self.env._get_observations(force_update=True) 
        ret = OrderedDict()
        if not self.low_dim:
            for k in self.keys:
                if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                    # by default images from mujoco are flipped in height
                    ret[k] = di[k][::-1]
                    if self.postprocess_visual_obs:
                        ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
                elif (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="depth"):
                    # by default depth images from mujoco are flipped in height
                    ret[k] = di[k][::-1]
                    if len(ret[k].shape) == 2:
                        ret[k] = ret[k][..., None] # (H, W, 1)
                    assert len(ret[k].shape) == 3 
                    # scale entries in depth map to correspond to real distance.
                    ret[k] = self.get_real_depth_map(ret[k])
                    if self.postprocess_visual_obs:
                        ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

        # "object" key contains object information
        ret["object"] =  np.array(di["object-state"]).flatten()
        
        

        
        for robot in self.env.robots:
            # add all robot-arm-specific observations. Note the (k not in ret) check
            # ensures that we don't accidentally add robot wrist images a second time
            pf = robot.robot_model.naming_prefix
            for k in di:
                if k.startswith(pf) and (k not in ret) and \
                    (not k.endswith("proprio-state")):
                    ret[k] =  np.array(di[k]).flatten()
                   
        
        return ret
        

    def get_flat_observation(self):
        return self._flatten_obs(self.get_observation())

    def get_real_depth_map(self, depth_map):
        
        # Make sure that depth values are normalized
        assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)
        extent = self.env.sim.model.stat.extent
        far = self.env.sim.model.vis.map.zfar * extent
        near = self.env.sim.model.vis.map.znear * extent
        return near / (1.0 - depth_map * (1.0 - near / far))



def load_initial(args): # returns robosuite env, inital local pose, reward functions, objects
    
    if (args.path_to_VLM_query is None):
        # use most recent VLM query
        directory = os.path.join(args.task_dir,'vlm_query')
        dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirs.sort()
        path_to_VLM_query = os.path.join(directory, dirs[-1])
    else:
        path_to_VLM_query = args.path_to_VLM_query
   
     
    
    env_meta = FileUtils.get_env_metadata_from_dataset(args.dataset)
    print('Loading robomimic env')
    env = EnvUtils.create_env_for_data_processing(
        env_meta = env_meta,
        camera_names= args.camera_names, 
        camera_height=args.camheight, 
        camera_width=args.camwidth, 
        reward_shaping=True,
        use_depth_obs=True, 
    )
    print('Robomimic env loaded')
    #print(env.__mro___)
    

    # load initial keypoint location in annotated environment so we can track their positions w.r.t poses 
    # in different poses.
    f = h5py.File(args.dataset, "r")    
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    ep = demos[0]
    states = f["data/{}/states".format(ep)][()]
    initial_state = dict(states=states[0])

    initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

    obs = env.reset()
 
    env.reset_to(initial_state)
    camera = args.camera_names[0]

    # get annotated keypoint poses so we can continue tracking them.

    keypoints_path = os.path.join(args.task_dir, 'annotations','first_frame_annotated.json')
    with open (keypoints_path) as f:
        keypoints=json.load(f)
    objects = get_objects_from_keypoints(keypoints)

    initial_local_pose = get_local_pose(env,
                                        keypoints,
                                        objects,
                                        camera,
                                        args.camwidth, 
                                        args.camheight)
    
    original_env = env.base_env
    reward_functions = get_reward_func(path_to_VLM_query)

    return original_env, initial_local_pose, reward_functions, objects
        




if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
        type=str, 
        required= True,
        help='Path to HDF5 dataset',
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        required=True,
        help="Folder to task, eg. ./data/can or ./data/cube",
    )
   
 
    parser.add_argument(
        '--path_to_VLM_query',
        type=str,
        default=None,
        help='(optional) Specify dir to VLM query. Will use latest generated if not provided.'
        
    )
    parser.add_argument(
        '--camera_names',
        type=str,
        nargs='+',
        default=['agentview']
    )

    parser.add_argument(
        '--camheight',
        type=int,
        default=1024
    )
    parser.add_argument(
        '--camwidth',
        type=int,
        default=1024
    )

    args = parser.parse_args()
    robomimicenv, initial_local_pose, reward_functions, objects = load_initial(args)
    env = CustomRewardGymWrapper(robomimicenv,
                                  objects, 
                                  initial_local_pose,
                                  reward_functions,
                                  True,
                                  'cpu', 
                                  args.camheight,
                                  args.camwidth,
                                  args.camera_names[0],
                                  )
  
    env.reset()
    
    check_env(env)
    
    
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    demos = demos[:1]
    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))
        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

    
    env.reset_to(initial_state)
    
    actions = f["data/demo_0/actions"][()]
    for i in range(len(actions)):
        obs,rew,done, info = env.step(actions[i])
     #   print(obs)
        print(f'Step {i} reward: {rew}')
    
 