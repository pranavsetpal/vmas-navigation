# Torch
import torch

# Env
from src.env import create_env
from src.models.hyperparameters import scenario_name, num_vmas_envs, max_steps, n_agents

# System
import sys
from os.path import isfile

# Check if file exists
policy_path = sys.argv[1]
if not isfile(policy_path):
    print("Error: {policy_path} does not exist", file=sys.stderr)
    sys.exit(1)

#Devices
# vmas_device = device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"
vmas_device = device = "cpu"

env = create_env(scenario_name, num_vmas_envs, max_steps, vmas_device, n_agents)
policy = torch.load(policy_path)

# Render policy
with torch.no_grad():
    env.rollout(
        policy=policy,
        max_steps=max_steps,
        callback=lambda env, _: env.render(),
        auto_cast_to_device=True,
        break_when_any_done=False
    )

