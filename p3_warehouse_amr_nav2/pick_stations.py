#!/usr/bin/env python3
"""
Records patrol stations by listening to RViz2's "Publish Point" clicks.
Click the stations IN PATROL ORDER. After NUM_STATIONS clicks, the
coordinates are saved to ~/p3_warehouse/config/stations.yaml.
"""
import os
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped   # message type RViz2 publishes on /clicked_point

NUM_STATIONS = 4
OUT_FILE = os.path.expanduser('~/p3_warehouse/config/stations.yaml')


class StationPicker(Node):
    def __init__(self):
        super().__init__('station_picker')
        self.stations = []
        # Every "Publish Point" click in RViz2 arrives here
        self.create_subscription(PointStamped, '/clicked_point', self.on_click, 10)
        self.get_logger().info(
            f'Click {NUM_STATIONS} stations with "Publish Point" in RViz2, in patrol order.')

    def on_click(self, msg: PointStamped):
        # Stations must be in the map frame, or Nav2 would misread them
        if msg.header.frame_id != 'map':
            self.get_logger().warn(
                f'Click was in frame "{msg.header.frame_id}". Set RViz2 Fixed Frame to "map".')
            return
        n = len(self.stations) + 1
        station = {
            'name': f'station_{n}',
            'x': round(msg.point.x, 2),   # metres, rounded to 1 cm
            'y': round(msg.point.y, 2),
        }
        self.stations.append(station)
        self.get_logger().info(f'Recorded {station["name"]}: x={station["x"]}, y={station["y"]}')


def main():
    rclpy.init()
    node = StationPicker()
    # Keep processing clicks until we have all the stations
    while rclpy.ok() and len(node.stations) < NUM_STATIONS:
        rclpy.spin_once(node, timeout_sec=0.5)

    # Save as YAML so the patrol script (and you) can read and edit it
    with open(OUT_FILE, 'w') as f:
        yaml.safe_dump({'stations': node.stations}, f, sort_keys=False)
    node.get_logger().info(f'Saved {len(node.stations)} stations to {OUT_FILE}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
