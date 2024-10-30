import torch
from robomimic.utils import file_utils as FileUtils
import robomimic
from rekep_rl.custom_gym_wrapper import load_initial, CustomRewardGymWrapper
 
import argparse

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
                                  'cpu', 
                                  args.camheight,
                                  args.camwidth,
                                  args.camera_names[0],
                                  )
    
    obs = env.reset()
    print(obs)
    o= env.get_observation()
    for key in o:
        print(f'{key} {o[key].shape}')