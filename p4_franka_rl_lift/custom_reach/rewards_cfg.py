"""Reward and curriculum design for Custom-Reach-Franka (Project 4, Milestone 3).

Design: "reach and hold"
  - coarse pull:  penalty proportional to distance         -> gets the hand close from far away
  - fine pull:    1 - tanh(distance / 0.1) bonus           -> rewards the last few centimetres
  - orientation:  penalty for rotation error               -> keeps the gripper pointing down
  - hold bonus:   paid every step inside the success zone  -> "stay there", not "touch and leave"
  - smoothness:   jerk and joint-speed penalties, ramped up by a curriculum
"""

import isaaclab.envs.mdp as mdp                               # ready-made reward and curriculum functions
from isaaclab.managers import CurriculumTermCfg as CurrTerm   # one curriculum rule
from isaaclab.managers import RewardTermCfg as RewTerm        # one reward term
from isaaclab.managers import SceneEntityCfg                  # selector: which asset and which bodies
from isaaclab.utils import configclass


@configclass
class CustomReachRewardsCfg:
    """Every term is computed each policy step, multiplied by its weight, and summed."""

    # Coarse pull: distance in metres. Strong signal far away, weak near the target.
    position_coarse = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]), "command_name": "ee_pose"},
    )

    # Fine pull: about 0 when far, rising to 1 within about std (0.1 m = 10 cm) of the target.
    position_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]), "std": 0.1, "command_name": "ee_pose"},
    )

    # Orientation: rotation error in radians between the hand and the (always pointing down) target.
    orientation = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]), "command_name": "ee_pose"},
    )

    # Hold bonus: 1 on every step the hand is within 5 cm AND 0.2 rad (the thresholds in CommandsCfg), else 0.
    hold_bonus = RewTerm(
        func=mdp.pose_command_success,
        weight=5.0,
        params={"command_name": "ee_pose"},
    )

    # Smoothness: penalize sudden changes in the action (jerk). Starts tiny; the curriculum ramps it up.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)

    # Calm motion: penalize joint speed. Starts tiny; the curriculum ramps it up.
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CustomReachCurriculumCfg:
    """Learn to reach first, then get smoother: the penalty weights grow over training."""

    # After 4500 env steps, raise the jerk penalty from -0.0001 to -0.005
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    # After 4500 env steps, raise the joint-speed penalty from -0.0001 to -0.001
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )
