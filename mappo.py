# Torch
import torch

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Data Collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm



#Devices
vmas_device = device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"

# Sampling
frames_per_batch = 6_000 # Number of team frames collected per training iteration
n_iters = 10 # Number of sampling and training itierations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 30
minibatch_size = 400
lr = 3e-4
max_grad_norm = 1.0 # Max norm for gradients

# PPO
clip_epsilon = 0.2 # max change between optimizations; clips loss
gamma = 0.9 # discount_factor
lmbda = 0.9 # lambda for generalised advantage estimation
entropy_eps = 1e-4 # coefficient of the entropy term in the PPO loss


## Environment
max_steps = 100
num_vmas_envs = frames_per_batch // max_steps
scenario_name = "navigation"
n_agents = 3

env = VmasEnv(
    scenario=scenario_name,
    num_envs=num_vmas_envs,
    countrinuous_actions=True,
    max_steps=max_steps,
    device=vmas_device,
    #Scenario kwargs
    n_agents=n_agents
)

# Transforms
env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")])
)


## Policy
share_parameters_policy = True
policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs = 2 * env.action_spec.shape[-1],
        n_agents=env.n_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh
    ),
    NormalParamExtractor() # Separarte last dimension into 2 outputs: loc, non-negative scale
)

policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "loc"), ("agents", "scale")]
)

policy = ProbabilisticActor(
    module=policy_module,
    spec=env.unbatched_action_spec,
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.unbatched_action_spec[env.action_key].space.low,
        "max": env.unbatched_action_spec[env.action_key].space.high
    },
    return_log_prob=True,
    log_prob_key=("agents", "sample_log_prob")
)


## Critic
share_parameters_critic = True

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1,
    n_agents=env.n_agents,
    centralised=True, # MAPPO
    share_params=share_parameters_critic, # Only if same rewards for all agents
    device=device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "state_value")]
)


## Data Management
collector = SyncDataCollector(
    env,
    policy,
    device=vmas_device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch, device=device), # STore fames_per_batch st each iter
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size
)


## Loss function
loss_module = ClipPPOLoss(
    actor=policy,
    critic=critic,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=False # IMP: Avoid normalizing across agent dim.
)
loss_module.set_keys(
    reward=env.reward_key,
    action=env.action_key,
    sample_log_prob=("agents", "sample_log_prob"),
    value=("agents", "state_value")
)

loss_module.make_value_estimator( ValueEstimators.GAE, gamma=gamma, lmbda=lmbda )
GAE = loss_module.value_estimator

optim = torch.optim.AdamW(loss_module.parameters(), lr)


## Render
# with torch.no_grad():
#     env.rollout(
#         max_steps=max_steps,
#         callback=lambda env, _: env.render(),
#         auto_cast_to_device=True,
#         break_when_any_done=False
#     )
