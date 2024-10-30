import torch
from robomimic.utils import file_utils as FileUtils
import robomimic
ckpt_path = '/nethome/atian31/flash8/repos/rekep_rl/bc/test/20241022172640/models/model_epoch_2000.pth'
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, verbose=True)
env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name='PickPlaceCan', 
        render=False, 
        render_offscreen=False, 
        verbose=False,
    )
o = env.reset()
for key in o:
        print(f'{key} {o[key].shape}')