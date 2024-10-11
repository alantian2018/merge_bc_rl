import os
import h5py
import argparse
import numpy as np
import robomimic
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import argparse

def get_robomimic_robosuite_env(args):
    env_meta = FileUtils.get_env_metadata_from_dataset(args.dataset)
    print('Loading robomimic env')
    robomimic_env = EnvUtils.create_env_for_data_processing(
        env_meta = env_meta,
        camera_names= args.camera_names, 
        camera_height=args.camheight, 
        camera_width=args.camwidth, 
        reward_shaping=True,
        use_depth_obs=True, 
    )
    print('Robomimic env loaded')
    robosuite_env = robomimic_env.base_env
    return robosuite_env,robomimic_env

def reset_robomimic_env(args, env):
    """
    Reset env to initial state at beginning of 0th robomimic trajectory. 
    """
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

def get_robosuite_objects(robosuite_env):
    """
    Returns a dictionary of {object1: {xpos, xquat}, object2: {xpos,xquat}, etc...}
    """
    objects = dict()
    for obj in robosuite_env.sim.model.body_names:
        l=[]
        l.append(robosuite_env.sim.data.get_body_xpos(obj))
        l.append(robosuite_env.sim.data.get_body_xquat(obj))
        objects[obj] = l

    return objects

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='path to hdf5 dataset file')
    parser.add_argument('--camera_names', nargs='+', default=['agentview'])
    parser.add_argument('--camheight', type=int, default = 512)
    parser.add_argument('--camwidth', type=int, default = 512)
    args = parser.parse_args()

    robosuite_env, robomimic_env = get_robomimic_robosuite_env(args)
    reset_robomimic_env(args, robomimic_env)

    # Get all objects in robosuite_env
    objects = get_robosuite_objects(robosuite_env)
    print(objects)