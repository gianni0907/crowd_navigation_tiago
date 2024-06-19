cmake_minimum_required(VERSION 3.0.2)
project(crowd_navigation_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  geometry_msgs
  std_msgs
  nav_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  CrowdMotionPrediction.msg
  CrowdMotionPredictionStamped.msg
  MotionPrediction.msg
  Measurement.msg
  MeasurementsSet.msg
  MeasurementsSetStamped.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  SetDesiredTargetPosition.srv
  SetDesiredHeadConfig.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES my_tiago_controller
  CATKIN_DEPENDS
    message_runtime
    geometry_msgs
    std_msgs
    nav_msgs
#  DEPENDS system_lib
)