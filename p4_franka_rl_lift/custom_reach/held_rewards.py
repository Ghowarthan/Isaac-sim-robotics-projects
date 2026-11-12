"""Held-gated lift rewards (Project 4, v9).

v5 learned to THROW the cube: its lift and goal rewards paid whenever the cube was high,
even in free flight (evaluation: peak 55 cm, only 0.64 s above 10 cm, matching a ballistic toss).
These versions pay only while the cube is IN THE HAND: lifted AND within `max_dist` of the
gripper's centre point (the TCP, measured by the ee_frame sensor).
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms  # converts the goal from robot frame to world frame

from isaaclab_tasks.contrib.lift import mdp  # classic lift functions (object_ee_distance, object_is_lifted, ...)


def cube_in_hand(env, max_dist: float = 0.04, minimal_height: float = 0.04) -> torch.Tensor:
    """Boolean per robot: cube lifted above `minimal_height` AND within `max_dist` of the TCP."""
    # object_ee_distance returns 1 - tanh(d / std). With std = 0.02 m:
    #   d < max_dist   <=>   value > 1 - tanh(max_dist / 0.02)
    # This reuses NVIDIA's own TCP-to-cube distance, so no new sensor code is needed.
    near_tcp = mdp.object_ee_distance(env, std=0.02) > (1.0 - math.tanh(max_dist / 0.02))
    lifted = mdp.object_is_lifted(env, minimal_height=minimal_height) > 0.5  # returns 1.0 / 0.0
    return near_tcp & lifted


def held_object_is_lifted(env, minimal_height: float = 0.04, max_dist: float = 0.04) -> torch.Tensor:
    """1 while the cube is lifted AND held, else 0 (replaces object_is_lifted)."""
    return cube_in_hand(env, max_dist, minimal_height).float()


class held_object_goal_distance(mdp.object_goal_distance):
    """Goal tracking (1 - tanh(distance / std)) paid ONLY while the cube is in the hand.

    Inherits the parent's per-episode success bookkeeping (Metrics/success_rate, logged at reset),
    but a success now requires the cube to be within `success_threshold` of the goal WHILE HELD,
    so a cube flying past the goal no longer counts.
    """

    def __call__(
        self,
        env,
        std: float,
        minimal_height: float,
        command_name: str,
        max_dist: float = 0.04,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        success_threshold: float | None = None,
    ) -> torch.Tensor:
        held = cube_in_hand(env, max_dist, minimal_height)            # in hand right now?
        robot = env.scene[robot_cfg.name]
        obj = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)      # goal position in the robot base frame
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
        )                                                            # goal position in the world frame
        distance = torch.linalg.norm(des_pos_w - obj.data.root_pos_w.torch, dim=1)
        if success_threshold is not None:
            self._succeeded |= held & (distance < success_threshold)  # sticky: a held delivery happened this episode
        return held.float() * (1 - torch.tanh(distance / std))


def held_lift_to_goal_height(
    env,
    command_name: str,
    minimal_height: float = 0.04,
    max_dist: float = 0.04,
    floor: float = 0.25,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """v10: held lift reward that grows with height toward the goal's height.

    v9 held the cube but kept it low (median peak 15.5 cm vs goals at 25-50 cm), because a flat
    lift reward pays the same at 5 cm as at 40 cm. Here:
        reward = held * (floor + (1 - floor) * progress)
        progress = (cube_z - minimal_height) / (goal_z - minimal_height), clipped to [0, 1]
    so holding still pays `floor`, but rising to the goal height pays up to 1 / floor times more.
    """
    held = cube_in_hand(env, max_dist, minimal_height)                    # in hand right now?
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)              # goal in the robot base frame
    goal_w, _ = combine_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
    )                                                                    # goal in the world frame
    cube_z = obj.data.root_pos_w.torch[:, 2]                             # cube centre height
    span = torch.clamp(goal_w[:, 2] - minimal_height, min=1e-3)          # avoid dividing by ~0
    progress = torch.clamp((cube_z - minimal_height) / span, 0.0, 1.0)   # 0 at 4 cm, 1 at goal height
    return held.float() * (floor + (1.0 - floor) * progress)
