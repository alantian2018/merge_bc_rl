## Attempt to merge BC and RL algorithms by loading BC policy into RL training loop

`./data/` directory contains data (VLM queries, reward functions, annotations, etc.)

`./merge_bc_rl/bc/20241006112611/models/model_epoch_2000.pth` contains a trained `bc-transformer-gmm` model. (BC) policy

The environment wrapper I am using during RL training can be found at `./rekep-rl/custom_gym_wrapper.py`

I've played around with a policy wrapper during training. See `train_rl.py` The changes I've made between this and the default rlkit experiment are<br>
- the custom wrapper for the policy at the top.<br>
- the loading the custom policy at line 83-89.

Right now, I am using the command 

```python train_rl.py --dataset [path to robomimic dataset] --task_dir [path to task reward functions, usually ./data/{task}] --path_to_policy [path to pretrained checkpoint]```

ex:
`python train_rl.py --dataset ./robomimic/data/square/ph/low_dim_v141.hdf5 --task_dir ./data/square --path_to_policy ./bc/20241006112611/models/model_epoch_2000.pth`

Appreciate your help!