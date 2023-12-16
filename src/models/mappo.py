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

# Env
from src.env import create_env

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import datetime

torch.manual_seed(0)

# Devices
# vmas_device = device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"
vmas_device = device = "cpu"

# Hyperparameters
from src.models.hyperparameters import frames_per_batch, n_iters, total_frames # Sampling
from src.models.hyperparameters import num_epochs, minibatch_size, lr, max_grad_norm # Training
from src.models.hyperparameters import clip_epsilon, gamma, lmbda, entropy_eps # PPO
from src.models.hyperparameters import max_steps, num_vmas_envs, scenario_name, n_agents # Environment


## Env
env = create_env(scenario_name, num_vmas_envs, max_steps, vmas_device, n_agents)

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


# Training loop
pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

episode_reward_mean_list = []
for tensordict_data in collector:
    tensordict_data.set(
        ("next", "done"),
        tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get(("next", env.reward_key)).shape), # Match reward shape
    )

    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_params,
            target_params=loss_module.target_critic_params
        )

    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)

    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(),
                max_grad_norm
            )
            optim.step()
            optim.zero_grad()

        collector.update_policy_weights_()

        # Logging
        done = tensordict_data.get(("next", "done"))
        episode_reward_mean = tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
        episode_reward_mean_list.append(episode_reward_mean)
        pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
        pbar.update()


## Results
plt.plot(episode_reward_mean_list)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean")
plt.show()


# Save model
torch.save(policy, f"./models/{datetime.now()}.pt")
