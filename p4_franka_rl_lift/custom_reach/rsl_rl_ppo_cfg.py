"""rsl_rl PPO training settings for Custom-Reach-Franka."""

from isaaclab.utils import configclass

# NVIDIA's PPO settings for Franka reach (network size, learning rate, steps per update, ...)
from isaaclab_tasks.core.reach.config.franka.agents.rsl_rl_ppo_cfg import FrankaReachPPORunnerCfg


@configclass
class CustomReachPPORunnerCfg(FrankaReachPPORunnerCfg):
    """Same PPO settings, our own log folder and run length."""

    def __post_init__(self):
        # Run the parent's own setup first, if it has one
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        # The parent defines these two as presets (default vs hold).
        # Replacing them with plain values, as NVIDIA's OSC config does, fixes them for our task.
        self.max_iterations = 1500                     # hold-style learning converged by about 500 iterations; 1500 leaves room
        self.experiment_name = "custom_reach_franka"   # logs go to logs/rsl_rl/custom_reach_franka
