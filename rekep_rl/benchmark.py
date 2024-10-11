import pickle as pkl
import numpy as np
import os
import matplotlib.pyplot as plt 
from rekep_rl.ReKep.utils import exec_safe
import argparse

def get_dense_reward_constraint(EE, keypoints, reward_functions):
    cost = 0 
    for func in reward_functions:
        cost -= func(EE, keypoints)
    return cost

def get_reward_func(path_to_VLM_query):
    
    list_of_reward_funcs = [i for i in os.listdir(path_to_VLM_query) 
                            if i.startswith('stage') and i.endswith('_subgoal_constraints.txt')]

    print(f'Loaded {len(list_of_reward_funcs)} reward functions')
    constraints = []
    for i in range (len(list_of_reward_funcs)):
        
        with open (os.path.join(path_to_VLM_query, f'stage{i+1}_subgoal_constraints.txt')) as f:
            function_text = f.read()
        gvars_dict = {'np': np}
        lvars_dict = dict()
        exec_safe(function_text, gvars=gvars_dict, lvars=lvars_dict)
        constraints += list(lvars_dict.values())
    return constraints


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--task_dir', required=True, type=str, help="Folder to task, eg. ./data/can or ./data/cube.")
    parser.add_argument('--vlm_query', default=None, type=str, help='(optional) path to vlm query. Default is most recently generated query')
    args = parser.parse_args()

    path_to_depth = os.path.join(args.task_dir, 'ground_truths','keypoints.pkl')
    
    save_path = os.path.join(os.path.dirname(path_to_depth), 'constraint_costs.png')

    with open (path_to_depth, 'rb' )as f:
        content = pkl.load(f)
        ee, depth = content['ee'], content['depth']
    
    if (args.vlm_query is None):
        directory = os.path.join(args.task_dir,'vlm_query')
        dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirs.sort()
    #    print(dirs)
        vlm_query = os.path.join(directory, dirs[-1])
    else:
        vlm_query = args.vlm_query
    
    reward_functions = get_reward_func(vlm_query)
    reward_function = get_dense_reward_constraint

    costs = []
    x=[]
    print('evaluating reward functions')
    for i, step in enumerate(zip(ee, depth)):
        c = reward_function(step[0],step[1], reward_functions)
        costs.append(c)
        x.append(i)
    plt.plot(x,costs)
    plt.title("Can costs evaluated on ReKep reward function")
    plt.xlabel='simulation step'
    plt.ylabel='cost'
    print('saving')
    plt.savefig(save_path)
    
    