"""
moveit_isaac.launch.py
Starts MoveIt 2 for the UR5e, with ros2_control talking to Isaac Sim
through /joint_command and /joint_states.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder   # Helper that gathers all MoveIt settings


def generate_launch_description():
    # Paths to our own package and the official UR MoveIt package
    pkg_share = get_package_share_directory("ur5e_isaac_moveit")
    ur_moveit_share = get_package_share_directory("ur_moveit_config")

    # ---------------- MoveIt configuration ----------------
    # Start from ur_moveit_config, then swap in our robot description and controller list
    moveit_config = (
        MoveItConfigsBuilder("ur5e", package_name="ur_moveit_config")
        # Our URDF: official UR5e + the ros2_control block for Isaac Sim
        .robot_description(
            file_path=os.path.join(pkg_share, "urdf", "ur5e_isaac.urdf.xacro"),
            mappings={"ur_type": "ur5e", "name": "ur5e"},
        )
        # SRDF = "semantic" description: planning group, self-collision rules, named poses
        .robot_description_semantic(file_path="srdf/ur.srdf.xacro", mappings={"name": "ur5e"})
        # Which IK solver to use for the arm
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        # Speed and acceleration limits used when timing a planned path
        .joint_limits(file_path="config/joint_limits.yaml")
        # Our phone book: which controller executes plans
        .trajectory_execution(file_path=os.path.join(pkg_share, "config", "moveit_controllers.yaml"))
        # OMPL = the sampling-based motion planner library
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    sim_time = {"use_sim_time": True}   # Every node follows Isaac Sim's /clock

    # 1. robot_state_publisher: joint angles -> TF
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description, sim_time],
        output="screen",
    )

    # 2. Controller manager with the TopicBasedSystem hardware plug
    ros2_control = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[os.path.join(pkg_share, "config", "ros2_controllers.yaml"), sim_time],
        # The controller manager reads the URDF (with our ros2_control block) from this topic
        remappings=[("~/robot_description", "/robot_description")],
        output="screen",
    )

    # 3. Start the joint trajectory controller inside the controller manager
    jtc_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    # 4. MoveIt's main node: IK, planning, execution
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        parameters=[moveit_config.to_dict(), sim_time, {"trajectory_execution.allowed_execution_duration_scaling": 3.0, "trajectory_execution.allowed_goal_duration_margin": 2.0}],
        output="screen",
    )

    # 5. RViz2 with the MotionPlanning panel (layout from ur_moveit_config)
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(ur_moveit_share, "config", "moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            sim_time,
        ],
        output="log",
    )

    return LaunchDescription([rsp, ros2_control, jtc_spawner, move_group, rviz])
