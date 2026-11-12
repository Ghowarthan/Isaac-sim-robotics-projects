#!/usr/bin/env python3
"""
Warehouse patrol: visits the stations in stations.yaml in a loop using
Nav2's Simple Commander API, printing live feedback and a summary.
Ctrl+C cancels the current goal cleanly and stops the robot.
"""
import math
import os
import time

import yaml
import rclpy
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions   # lets us catch Ctrl+C ourselves
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

STATIONS_FILE = os.path.expanduser('~/p3_warehouse/config/stations.yaml')
LAPS = 2          # how many times to go around the full route
DWELL_S = 2.0     # pause at each station (seconds), like stopping to "load"


def make_pose(nav: BasicNavigator, x: float, y: float, yaw: float) -> PoseStamped:
    """Build a Nav2 goal: position (x, y) and heading yaw (radians) in the map frame."""
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = nav.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    # Heading as a quaternion: a rotation of yaw about the vertical (z) axis
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose


def main():
    # Turn off rclpy's own Ctrl+C handler so we can cancel the goal before exiting
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    nav = BasicNavigator(node_name='warehouse_patrol')
    nav.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])  # follow Isaac Sim's clock
    log = nav.get_logger()

    with open(STATIONS_FILE) as f:
        stations = yaml.safe_load(f)['stations']
    log.info(f'Loaded {len(stations)} stations: {[s["name"] for s in stations]}')

    # Wait for Nav2 to be active. localizer='bt_navigator' skips the AMCL wait on purpose:
    # that wait publishes a default (0, 0) initial pose, which would overwrite the
    # pose AMCL already has.
    nav.waitUntilNav2Active(localizer='bt_navigator')
    log.info('Nav2 is active. Starting patrol.')

    succeeded, failed = 0, 0
    try:
        for lap in range(1, LAPS + 1):
            for i, st in enumerate(stations):
                # Face the NEXT station on arrival, so the robot leaves without spinning
                nxt = stations[(i + 1) % len(stations)]
                yaw = math.atan2(nxt['y'] - st['y'], nxt['x'] - st['x'])

                log.info(f'[Lap {lap}/{LAPS}] Heading to {st["name"]} '
                         f'(x={st["x"]}, y={st["y"]}, facing {math.degrees(yaw):.0f} deg)')
                nav.goToPose(make_pose(nav, st['x'], st['y'], yaw))   # send the action goal

                # Feedback loop: isTaskComplete() also processes incoming messages
                last_print = 0.0
                while not nav.isTaskComplete():
                    fb = nav.getFeedback()
                    if fb and time.time() - last_print > 2.0:   # print every 2 s, not every tick
                        log.info(f'    {fb.distance_remaining:5.1f} m left, '
                                 f'ETA {fb.estimated_time_remaining.sec:3d} s, '
                                 f'recoveries: {fb.number_of_recoveries}')
                        last_print = time.time()

                result = nav.getResult()   # the action's final result
                if result == TaskResult.SUCCEEDED:
                    succeeded += 1
                    log.info(f'    Arrived at {st["name"]}. Dwelling {DWELL_S:.0f} s.')
                    time.sleep(DWELL_S)
                elif result == TaskResult.CANCELED:
                    log.warn('    Goal was canceled. Stopping patrol.')
                    return
                else:   # FAILED: log it and move on, like a real patrol would
                    failed += 1
                    log.error(f'    Could not reach {st["name"]}. Skipping to the next station.')

    except KeyboardInterrupt:
        log.warn('Ctrl+C: canceling the current goal so the robot stops.')
        nav.cancelTask()

    finally:
        log.info(f'Patrol summary: {succeeded} reached, {failed} failed.')
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
