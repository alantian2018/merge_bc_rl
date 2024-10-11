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

from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
import robomimic.utils.file_utils as FileUtils

from rekep_rl.custom_gym_wrapper import load_initial, CustomRewardGymWrapper

class robomimic_policy:
    def __init__(self, policy):
        self.policy = policy

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def parameters(self):
        return self.policy.nets['policy'].parameters()

    def to(self, device):
        self.policy.nets['policy'].to(device)
 
    def reset(self):
        pass
        
    def close(self):
        pass


def experiment(variant,expl_env,eval_env, path_to_policy, device):
    
    obs_dim = expl_env.observation_space.low.size
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
    else:
        print(f"Warm starting RL training with model at {path_to_policy}")
        Rolloutpolicy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=path_to_policy, device=device, verbose=True)
        policy = Rolloutpolicy.policy 
         
        policy = robomimic_policy(policy)

      #  print([i for i in policy.parameters() if i.requires_grad])
        

    eval_policy = MakeDeterministic(policy)
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
    algorithm.train()






if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=2000,
            num_eval_steps_per_epoch=4000,
            num_trains_per_train_loop=800,
            num_expl_steps_per_train_loop=800,
            min_num_steps_before_training=800,
            expl_max_path_length=400,
            eval_max_path_length=400,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
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

    robomimicenv, initial_local_pose, reward_functions, objects = load_initial(args)
    expl_env = NormalizedBoxEnv(CustomRewardGymWrapper(
                                  robomimicenv,
                                  objects,
                                  initial_local_pose,
                                  reward_functions,
                                  args.camheight,
                                  args.camwidth,
                                  args.camera_names[0]))

    eval_env = NormalizedBoxEnv(CustomRewardGymWrapper(
                                  robomimicenv,
                                  objects,
                                  initial_local_pose,
                                  reward_functions,
                                  args.camheight,
                                  args.camwidth,
                                  args.camera_names[0]))


    setup_logger('square-rl', variant=variant)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ptu.set_gpu_mode(device=='cuda')  # optionally set the GPU (default=False)
    print('starting experiment....')
    experiment(variant,expl_env,eval_env, args.path_to_policy, device)