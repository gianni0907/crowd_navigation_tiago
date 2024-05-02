cmake_minimum_required(VERSION 3.0.2)
project(crowd_navigation_core)

find_package(catkin REQUIRED COMPONENTS
  rospy
  crowd_navigation_msgs
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES crowd_navigation_core
  CATKIN_DEPENDS
    rospy
    crowd_navigation_msgs
#  DEPENDS system_lib
)
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/agents_detection
  scripts/motion_generation
  scripts/crowd_prediction
  scripts/plotter
  scripts/send_desired_target_position
  scripts/send_actors_trajectory
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
