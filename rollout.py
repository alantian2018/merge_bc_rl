
from util_rlkit.rlkit_utils import simulate_policy
 
import numpy as np
import torch
import imageio
import os
import json

from signal import signal, SIGINT
from sys import exit

from train_rl import *  
import argparse
from rekep_rl.custom_gym_wrapper import load_initial, CustomRewardGymWrapper


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
        '--path_to_policy',
        type=str,
        required=True,
        help='Path to policy for rolling out'
        
        )
    parser.add_argument(
        '--path_to_warmstart',
        type=str,
        default=None
    )
    parser.add_argument(
        '--path_to_VLM_query',
        type=str,
        default=None
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

  
    
    robomimicenv, initial_local_pose, reward_functions, objects = load_initial(args)
    env = CustomRewardGymWrapper(
                                  robomimicenv,
                                  objects,
                                  initial_local_pose,
                                  reward_functions,
                                  args.camheight,
                                  args.camwidth,
                                  args.camera_names[0])

    video_writer = imageio.get_writer(os.path.join(args.path_to_policy, f'rl_policy_rollout.mp4'), fps=20)
    map_location = torch.device("cuda") if True else torch.device("cpu")
    data = torch.load(os.path.join(args.path_to_policy, "params.pkl"), map_location=map_location)
    policy = data['evaluation/policy']

    if args.path_to_warmstart is not None:
        
        Rolloutpolicy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.path_to_warmstart, verbose=True)
        policy = Rolloutpolicy.policy 
        policy = robomimic_policy(policy, data['evaluation/modality_dims'])
        print(type(data['evaluation/policy'][0]))
        policy.update_weights_from_ckpt(data['evaluation/policy'][0])

    print(policy)

    policy.eval()

    simulate_policy(
        env=env,
        policy=policy,
        horizon=200,
        render=False,
        video_writer=video_writer,
        num_episodes=3,
        printout=True,
      
    )
