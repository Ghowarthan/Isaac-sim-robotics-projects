"""rsl_rl PPO training settings for Custom-Lift-Franka.

v1-v3: NVIDIA's settings (init_std 1.0, entropy 0.006) -> noise grew to 1.3-1.4, too shaky to grasp
v4:    init_std 0.5, entropy 0.001 -> grasp + lift, but noise collapsed to 0.09 (never explored carrying)
v5:    entropy 0.003 -> carrying learned, 25% success
v7:    + observation normalization (actor and critic) + 3000 iterations
"""

from isaaclab.utils import configclass

# NVIDIA's PPO settings for the classic cube lift (network 256-128-64, adaptive learning rate)
from isaaclab_tasks.contrib.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg


@configclass
class CustomLiftPPORunnerCfg(LiftCubePPORunnerCfg):
    """NVIDIA's lift PPO settings with tuned exploration and normalization, saved to our own log folder."""

    def __post_init__(self):
        # Run the parent's own setup first, if it has one
        parent_post_init = getattr(super(), "__post_init__", None)
        if parent_post_init is not None:
            parent_post_init()
        self.experiment_name = "custom_lift_franka"  # logs go to logs/rsl_rl/custom_lift_franka

        # v7: twice as long (v5 was still creeping upward at 1500)
        self.max_iterations = 2000

        # v4/v5 exploration: half the starting noise, moderate "stay random" bonus
        self.actor.distribution_cfg.init_std = 0.5
        self.algorithm.entropy_coef = 0.003

        # v7: normalize observations (joint angles in rad, positions in m and gripper state have very different scales)
        self.actor.obs_normalization = False  # v9: off, as in v5
        self.critic.obs_normalization = False  # v9: off, as in v5
