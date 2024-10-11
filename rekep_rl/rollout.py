
from util_rlkit.rlkit_utils import simulate_policy
 
import numpy as np
import torch
import imageio
import os
import json

from signal import signal, SIGINT
from sys import exit

  
import argparse
 
from custom_gym_wrapper import *
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
 

 
 
 
if __name__ == "__main__":
    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)

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
        '--camwidth',
        type=int,
        default=1024
    )
    parser.add_argument(
        '--camheight',
        type=int,
        default=1024
    )
    args = parser.parse_args()

    path_to_policy = '/nethome/atian31/flash8/repos/rlkit/data/square-rl/square-rl_2024_10_08_12_57_17_0000--s-0'
    
    robomimicenv, initial_local_pose, reward_functions, objects = load_initial(args)
    env = CustomRewardGymWrapper(
                                  robomimicenv,
                                  objects,
                                  initial_local_pose,
                                  reward_functions,
                                  args.camheight,
                                  args.camwidth,
                                  args.camera_names[0])

    video_writer = imageio.get_writer(os.path.join(args.task_dir, f'policy_rollout.mp4'), fps=20)
    simulate_policy(
        env=env,
        model_path=os.path.join(path_to_policy, "params.pkl"),
        horizon=100,
        render=False,
        video_writer=video_writer,
        num_episodes=3,
        printout=True,
        use_gpu=True,
    )
