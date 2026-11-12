"""Custom reward functions for Custom-Lift-Franka (Project 4, Milestone 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:  # type hints only; not imported at runtime
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def grasp_at_object(
    env: ManagerBasedRLEnv,
    std: float,
    action_name: str = "gripper_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing the gripper while the TCP is at the cube.

    Returns one value per robot, shape (num_envs,):
        closeness x is_closing, where closeness = 1 - tanh(distance / std),
        so it is about 1 only when the fingers are commanded closed with the TCP right at the cube.
    """
    # Look up the cube and the TCP sensor in the scene
    cube: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Cube centre in world coordinates: (num_envs, 3). ".torch" converts the simulator's Warp array to a PyTorch tensor.
    cube_pos_w = cube.data.root_pos_w.torch
    # TCP (point between the fingertips, 0.1034 m beyond panda_hand) in world coordinates: (num_envs, 3)
    tcp_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]

    # Straight-line TCP-to-cube distance per robot: (num_envs,)
    distance = torch.linalg.norm(cube_pos_w - tcp_pos_w, dim=1)
    # Smooth closeness score: about 1 within ~std of the cube, falling towards 0 further away
    closeness = 1.0 - torch.tanh(distance / std)

    # Gripper command from the policy: (num_envs, 1). Binary action: negative = close, positive = open.
    gripper_cmd = env.action_manager.get_term(action_name).raw_actions
    is_closing = (gripper_cmd[:, 0] < 0.0).float()

    # Pays only for closing AT the cube; closing anywhere else earns nothing
    return closeness * is_closing
