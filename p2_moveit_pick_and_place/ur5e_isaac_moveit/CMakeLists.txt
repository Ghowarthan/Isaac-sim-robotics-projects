cmake_minimum_required(VERSION 3.8)
project(ur5e_isaac_moveit)

# ament_cmake is the ROS 2 build system for CMake packages
find_package(ament_cmake REQUIRED)

# Copy our config, launch and urdf folders into the install space,
# so ROS can find them by package name
install(DIRECTORY config launch urdf
  DESTINATION share/${PROJECT_NAME})

ament_package()
