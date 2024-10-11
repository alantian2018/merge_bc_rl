import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import torch
import json
from ReKep.utils import *
from ReKep.constraint_generation import *
from ReKep.get_3d_points import get_world_coords
import matplotlib.pyplot as plt
import argparse


class ManuallyAnnotatedConstraints:
    def __init__(self, path_to_data,  visualize=False):
        global_config = get_config(config_path="./ReKep/configs/config.yaml")
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
       
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'], save_dir = os.path.dirname(
                                                                                                          path_to_data))
        self.path_to_data = path_to_data

    def generate_constraints(self, task, instr):
 
        PATH_TO_ANNOTATED = os.path.join(self.path_to_data, 'annotated_images/annotated_first_frame.jpg')
    #    print(PATH_TO_ANNOTATED)
        projected_img = cv2.imread(PATH_TO_ANNOTATED)[...,::-1]
     
        print(f"Querying VLM for {task}")
        self.constraint_generator.generate(projected_img, instr)

    

    def benchmark(self, path_to_final_reward_function, task):
        pre_task_path = os.path.join(self.path_to_data ,f'{task}_pre')
        post_task_path = os.path.join(self.path_to_data ,f'{task}_post')
        
 
        # load 2d pixel -> 3d world coordinates for pre_task and post_task
        pre_keypoints = self._load_global(pre_task_path)
        post_keypoints = self._load_global(post_task_path)
       # print(pre_coords,post_coords)


        self._add_reward_function(path_to_final_reward_function)
        
        initial_distance = self._run_reward_func(pre_keypoints)
        final_distance = self._run_reward_func(post_keypoints)

         
        print(f"=============== TASK {task} ===============")
        print(f"INITAL CONSTRAINT {initial_distance}")
        print(f"FINAL CONSTRAINT {final_distance}")
        
        if (task =='ranch'):
            mid_task_path = os.path.join(self.path_to_data ,f'{task}_mid')
            mid_keypoints = self._load_global(mid_task_path)
            mid_distance = self._run_reward_func(mid_keypoints)
            return initial_distance, mid_distance, final_distance

        return initial_distance, final_distance

    def _run_reward_func(self,keypoints):
        cost = 0
        for i,func in enumerate(self.reward_function):
            stage_cost = func(None,keypoints)
         #   print(stage_cost)
            cost += stage_cost    
        return cost

    def _add_reward_function(self, path):
        with open (path) as f:
            function_text = f.read()
         
        gvars_dict = {'np': np}
        lvars_dict = dict()
        exec_safe(function_text, gvars=gvars_dict, lvars=lvars_dict)
        self.reward_function = list(lvars_dict.values())
    
    def _load_global(self, path):
        with open (os.path.join(path, 'annotated.json')) as f:
            keypoints = json.load(f)['keypoints']
        pts = get_world_coords(path)
        return self._transform_coords(keypoints,pts)
        

    def _transform_coords(self,keypoints , point_cloud):
        new_keypoints = []
        for kp in keypoints:
            x,y = kp
            if (x==-1 or y==-1):
                new_keypoints.append([-1,-1,-1])
            else:
                new_keypoints.append(point_cloud[x,y])
        return np.array(new_keypoints)

def plot_differences(results):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for index, task  in enumerate(results):
        ax = axs.flat[index]
        if (task == 'ranch'):
           
            x_values = [0, 1, 2]  # Positions for the bars
            y_values = results[task]  # The pair of integers
            # Plot bar chart for the pair
            ax.bar(x_values, y_values,   color=['orange', 'red','blue'])
            ax.set_xticks(x_values)
            ax.set_xticklabels(['Inital','Mid', 'Final']) 
        else:
            # Extract the pair
            x_values = [0, 1]  # Positions for the bars
            y_values = results[task]  # The pair of integers
            # Plot bar chart for the pair
            ax.bar(x_values, y_values,   color=['orange', 'red'])
            ax.set_xticks(x_values)
            ax.set_xticklabels(['Inital', 'Final'])
        ax.set_title(task)
        
        ax.set_ylim(min(0, min(y_values))*1.03, max(y_values) *1.03)
    plt.tight_layout()
    plt.savefig('ReKep-comparison.png')
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--instruction', type=str, required=True, help='instruction for task')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    path = os.path.join('./data',args.task, 'annotations')
    
    run = ManuallyAnnotatedConstraints(path)
   
    run.generate_constraints(args.task, args.instruction)

  