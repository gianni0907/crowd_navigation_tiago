cmake_minimum_required(VERSION 3.0.2)
project(my_tiago_controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  my_tiago_msgs
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES my_tiago_controller
  CATKIN_DEPENDS
    rospy
    geometry_msgs
    nav_msgs
    my_tiago_msgs
#  DEPENDS system_lib
)
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/nmpc_controller
  scripts/logger
  scripts/plotter
  scripts/send_desired_target_position
  scripts/send_actors_trajectory
  scripts/crowd_prediction
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
