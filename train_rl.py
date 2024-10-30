import os
import torch
import argparse
import numpy as np

from util_rlkit.rlkit_custom import rollout
from util_rlkit.rlkit_custom import CustomTorchBatchRLAlgorithm

from rlkit.core import logger
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.launchers.launcher_util import setup_logger
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
import rlkit.torch.pytorch_util as ptu
from collections import OrderedDict
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
import robomimic.utils.file_utils as FileUtils

from rekep_rl.custom_gym_wrapper import load_initial, CustomRewardGymWrapper

class robomimic_policy:
    def __init__(self, policy, obs_modality_dims):
        
        self.policy = policy
        self.model = self.policy.nets['policy']
        self.modality_dims = obs_modality_dims
        
        assert isinstance(self.modality_dims, OrderedDict)
        self.key = []
        self.key_size = []
        for key,dim in self.modality_dims.items():
            self.key.append(key)
            self.key_size.append(dim)

    def __getattr__(self, name):
        try:
            return  getattr(self.policy, name)
        except:
            return getattr(self.model, name)

    def __call__(self, obs):
        if not isinstance(obs, OrderedDict):
            o = OrderedDict()
            
            t = obs.split(self.key_size, dim = 1)
           

            for key, key_obs in zip(self.key , t):
                o[key] = torch.unsqueeze(key_obs,1)
            obs = o
        assert isinstance(obs, OrderedDict)
        out = self.model.forward_train(obs)
        
        return out

    def reset(self):
        pass
        
    def close(self):
        pass
 
    def state_dict(self):
        return self.model.state_dict(),
  
    def update_weights_from_ckpt(self, PATH):
        self.model.load_state_dict(PATH) 
        self.model.eval()

    def get_action(self, obs):
        if not isinstance(obs, OrderedDict):
            o = OrderedDict()
            obs = torch.from_numpy(obs)
             
            t = obs.split(self.key_size, dim = 0)
           

            for key, key_obs in zip(self.key , t):
                o[key] = torch.reshape(key_obs,(1,-1)).float()
                o[key]  = o[key].to('cuda:0')
            obs = o
       # print(obs)
        a= self.policy.get_action(obs)
        a = a.flatten()
        
        return a.cpu().numpy()
        

def experiment(variant,expl_env,eval_env, path_to_policy, device):
    
    obs_dim = expl_env.obs_dim
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
  
    if not path_to_policy:
        print('No warm start model detected, training without warm start')    
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M],
        )
      #  print(policy)
    else:
        print(f"Warm starting RL training with model at {path_to_policy}")
        Rolloutpolicy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=path_to_policy, device=device, verbose=True)
        policy = Rolloutpolicy.policy 
        policy = robomimic_policy(policy, expl_env.modality_dims)
       

      #  print([i for i in policy.parameters() if i.requires_grad])
        

    eval_policy = policy #MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = CustomTorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    print('training...')
    algorithm.train()






if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1e6),
        algorithm_kwargs=dict(
            num_epochs=2,
            num_eval_steps_per_epoch=2000,
            num_trains_per_train_loop=800,
            num_expl_steps_per_train_loop=800,
            min_num_steps_before_training=800,
            expl_max_path_length=250,
            eval_max_path_length=250,
            batch_size=256,
            frozen_policy_epochs=100
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=1E-4,
            qf_lr=1E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            
        ),
    )


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
        "--path_to_policy",
        type=str,
        required=False,
        help='If warm starting, path to warm started model'
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

    # do NOT flatten if using robomimic forward calls and thus must match their obs convention
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    robomimicenv, initial_local_pose, reward_functions, objects = load_initial(args)
    expl_env = CustomRewardGymWrapper(
                                  robomimicenv,
                                  objects,
                                  initial_local_pose,
                                  reward_functions,
                                  args.camheight,
                                  args.camwidth,
                                  args.camera_names[0])

    eval_env = CustomRewardGymWrapper(
                                  robomimicenv,
                                  objects,
                                  initial_local_pose,
                                  reward_functions,
                                  args.camheight,
                                  args.camwidth,
                                  args.camera_names[0])


    setup_logger(args.task_dir.split( '/')[-1], variant=variant)
    
    ptu.set_gpu_mode(device=='cuda')  # optionally set the GPU (default=False)
    print('starting experiment....')
    experiment(variant,expl_env,eval_env, args.path_to_policy, device)