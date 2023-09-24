# env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

def create_env(scenario_name, num_vmas_envs, max_steps, vmas_device, n_agents):
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

    check_env_specs(env)

    return env
