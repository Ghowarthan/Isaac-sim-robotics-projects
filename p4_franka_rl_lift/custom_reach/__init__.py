"""Custom Franka environments (Project 4): reach and lift."""

import gymnasium as gym  # Gymnasium keeps the global list of task names

# Milestones 2-3: reach and hold
gym.register(
    id="Custom-Reach-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # the generic manager-based engine that reads our config
    disable_env_checker=True,                       # skip gym's single-env checks (Isaac Lab runs thousands at once)
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_reach_env_cfg:CustomReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:CustomReachPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# Milestone 4: cube lift
gym.register(
    id="Custom-Lift-Franka",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_lift_env_cfg:CustomLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_lift_ppo_cfg:CustomLiftPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
