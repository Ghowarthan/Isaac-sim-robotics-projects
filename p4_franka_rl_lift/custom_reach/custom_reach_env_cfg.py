"""Custom Franka reach environment, written from scratch (Project 4, Milestone 2).

Task: move the Franka's hand (panda_hand) to a randomly placed target point,
with the gripper always pointing straight down.
"""

import math  # math.pi = 180 degrees in radians

import isaaclab.envs.mdp as mdp                                # library of ready-made MDP functions (observations, rewards, resets)
import isaaclab.sim as sim_utils                               # spawners for shapes, ground planes and lights
from isaaclab.assets import ArticulationCfg, AssetBaseCfg      # config types: robots (articulations) and static objects
from isaaclab.envs import ManagerBasedRLEnvCfg                 # base class for a manager-based RL environment
from isaaclab.managers import EventTermCfg as EventTerm        # one reset or randomization event
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # a group of observations fed to one network
from isaaclab.managers import ObservationTermCfg as ObsTerm    # one observation term
from isaaclab.managers import RewardTermCfg as RewTerm         # one reward term
from isaaclab.managers import SceneEntityCfg                   # selector: "which asset, which bodies"
from isaaclab.managers import TerminationTermCfg as DoneTerm   # one episode-ending rule
from isaaclab.scene import InteractiveSceneCfg                 # base class for a scene that is cloned per environment
from isaaclab.utils import configclass, replace                # @configclass decorator; replace() copies a config with changes
from isaaclab.utils.noise import UniformNoiseCfg as Unoise     # uniform sensor noise
from isaaclab.visualizers import VisualizerCfg                 # default viewport camera position

from isaaclab_tasks.core.reach.reach_env_cfg import ReachPhysicsCfg  # NVIDIA's physics presets, so physics=isaacsim_physx keeps working

from .rewards_cfg import CustomReachCurriculumCfg, CustomReachRewardsCfg  # Milestone 3 design

from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip    # ready-made Franka Panda config (USD model + joint drives)


##
# Scene: the cell layout. Built once in env_0, then cloned to every environment.
##


@configclass
class CustomReachSceneCfg(InteractiveSceneCfg):
    """Ground, table, robot and light."""

    # Ground plane, just below the bottom of the table
    ground = AssetBaseCfg(
        prim_path="/World/ground",  # one shared ground for all environments (no {ENV_REGEX_NS})
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Table: a 0.9 x 1.3 x 1.0 m box centred at z = -0.5, so its top surface is at z = 0 where the robot stands
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",  # {ENV_REGEX_NS} means "inside each env_N", so every environment gets a table
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 1.3, 1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),  # make the table solid
        ),
    )

    # Robot: a copy of NVIDIA's Franka config, placed inside each environment
    robot: ArticulationCfg = replace(FRANKA_PANDA_CFG, prim_path="{ENV_REGEX_NS}/Robot")

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# Commands: the work order (where the hand should go).
##


@configclass
class CommandsCfg:
    """Target pose for the hand."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",              # the body whose pose is compared with the target
        resampling_time_range=(4.0, 4.0),    # draw a new target every 4 s
        debug_vis=True,                      # show the target frame (and the hand frame) in the viewport
        position_success_threshold=0.05,     # "reached" = within 5 cm (used only if we add a success rule in Milestone 3)
        orientation_success_threshold=0.2,   # ... and within 0.2 rad
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),              # metres in front of the robot base
            pos_y=(-0.2, 0.2),               # metres to the left and right
            pos_z=(0.15, 0.5),               # metres above the table top
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),        # rotate 180 degrees so the gripper points down
            yaw=(0.0, 0.0),                  # fixed yaw: our simplification (NVIDIA randomizes this)
        ),
    )


##
# Actions: the actuators.
##


@configclass
class ActionsCfg:
    """7 joint position targets for the arm."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],  # matches panda_joint1..7; the fingers are panda_finger_joint*, so they are excluded
        scale=0.5,                      # a policy output of 1.0 means a 0.5 rad offset
        use_default_offset=True,        # offsets are measured from the robot's default pose, not from zero
    )


##
# Observations: the sensors and gauges.
##


@configclass
class ObservationsCfg:
    """What the policy sees each step."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy network (order is preserved)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))  # 9 joint angles, relative to default, with noise
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))  # 9 joint speeds, with noise
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})  # target pose: 3 position + 4 quaternion
        actions = ObsTerm(func=mdp.last_action)  # the 7 actions from the previous step

        def __post_init__(self):
            self.enable_corruption = True   # switch the noise on
            self.concatenate_terms = True   # join all terms into one vector: 9 + 9 + 7 + 7 = 32

    policy: PolicyCfg = PolicyCfg()


##
# Events: changeover and reset.
##


@configclass
class EventCfg:
    """What happens when an episode resets."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",                         # run at every episode reset
        params={
            "position_range": (0.5, 1.5),     # start joints at 50-150% of their default angles
            "velocity_range": (0.0, 0.0),     # start at rest
        },
    )


##
# Terminations: end-of-cycle rules.
##


@configclass
class TerminationsCfg:
    """For now, episodes end only on the timer. Whether success ends them is a Milestone 3 decision."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# The full environment: all blocks brought together.
##


@configclass
class CustomReachEnvCfg(ManagerBasedRLEnvCfg):
    """Custom Franka reach environment."""

    scene: CustomReachSceneCfg = CustomReachSceneCfg(num_envs=4096, env_spacing=2.5)  # 4096 cells, 2.5 m apart
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: CustomReachRewardsCfg = CustomReachRewardsCfg()  # Milestone 3 reward design
    curriculum: CustomReachCurriculumCfg = CustomReachCurriculumCfg()  # ramps up the smoothness penalties
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Timing, physics and camera."""
        self.decimation = 4                     # the policy acts once every 4 physics steps
        self.episode_length_s = 12.0            # 12 s episodes = 360 policy steps
        self.sim.dt = 1 / 120                   # physics at 120 Hz, so the policy runs at 30 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics = ReachPhysicsCfg()    # PhysX / Newton presets, chosen with physics=... on the command line
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(3.5, 3.5, 3.5))  # starting camera position
