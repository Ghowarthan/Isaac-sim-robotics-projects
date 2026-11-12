"""Custom Franka cube-lift environment (Project 4, Milestone 4).

History:
v1 classic reward + cube spawned at resting height  -> flick hack, then lazy collapse
v2 + drop penalty, gentler curriculum                -> no cheating, but never grasped
v3 + custom grasp reward                             -> still no grasp (too shaky)
v4 + calmer exploration (PPO settings)               -> grasp + lift, but only held the cube
v5 + a little more exploration                       -> carries to goal, 25% success, plateau
v6 v5 + 8 s episodes only                            -> WORSE: 15% success, drops 7% -> 24%. "Not enough time" rejected.
v7 v5 + cheaper speed penalty + mid-range goal term (+ obs normalization, 3000 iters in PPO settings); 5 s episodes
"""

from isaaclab.managers import RewardTermCfg as RewTerm   # one reward term
from isaaclab.managers import SceneEntityCfg             # selector: which asset
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.lift import mdp  # classic lift functions: object_ee_distance, object_is_lifted, object_goal_distance, ...

from .lift_rewards import grasp_at_object  # your custom grasp reward (v3)

# NVIDIA's classic Franka lift: table, cube, TCP sensor (ee_frame), 7 arm joints + binary gripper, commands, observations
from isaaclab_tasks.contrib.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg


@configclass
class CustomLiftRewardsCfg:
    """Staged reward: reach -> grasp -> lift -> carry (coarse, mid, fine), plus a penalty for losing the cube."""

    # Stage 1, reach: 1 - tanh(TCP-to-cube distance / 0.1)
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # Stage 2, grasp (v3): fingers commanded closed with the TCP within about 2 cm of the cube
    grasp_at_object = RewTerm(func=grasp_at_object, params={"std": 0.02}, weight=3.0)

    # Stage 3, lift: 1 when the cube's centre is above 4 cm (world z)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)

    # Stage 4, carry (coarse, std 30 cm): pull from far away. Only while lifted. Also logs Metrics/success_rate (within 5 cm).
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose", "success_threshold": 0.05},
        weight=16.0,
    )

    # Stage 4, carry (mid, std 10 cm), NEW in v7: bridges the gap between ~23 cm (where v5 stalled) and 5 cm
    object_goal_tracking_mid = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.1, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=8.0,
    )

    # Stage 4, carry (fine, std 5 cm): rewards the last few centimetres
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # Safety (v2): one drop costs 500 x 0.02 s = 10 reward units
    drop_penalty = RewTerm(func=mdp.is_terminated_term, weight=-500.0, params={"term_keys": ["object_dropping"]})

    # Smoothness penalties, ramped by the curriculum (targets set in __post_init__ below)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class CustomLiftEnvCfg(FrankaCubeLiftEnvCfg):
    """Classic Franka lift scene with our own reward class. Episodes stay at the default 5 s (v6 showed 8 s is worse)."""

    rewards: CustomLiftRewardsCfg = CustomLiftRewardsCfg()

    def __post_init__(self):
        super().__post_init__()  # builds NVIDIA's scene, actions, cube, TCP sensor and curriculum first

        # v1: spawn the cube at about its resting height (no free "lifted" points while it falls)
        self.scene.object.init_state.pos = (0.5, 0.0, 0.03)

        # v2: smoothness curriculum ramps to -0.01 (NVIDIA: -0.1, which caused the lazy collapse)
        self.curriculum.action_rate.params["weight"] = -0.01
        # v7: speed penalty target -0.01 -> -0.001 (it was the biggest penalty in v5)
        self.curriculum.joint_vel.params["weight"] = -0.001
