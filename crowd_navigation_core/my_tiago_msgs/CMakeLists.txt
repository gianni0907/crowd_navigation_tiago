cmake_minimum_required(VERSION 3.0.2)
project(my_tiago_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  geometry_msgs
  std_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  CrowdMotionPrediction.msg
  CrowdMotionPredictionStamped.msg
  MotionPrediction.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  SetDesiredTargetPosition.srv
  SetActorsTrajectory.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs  # Or other packages containing msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES my_tiago_controller
  CATKIN_DEPENDS
    message_runtime
    geometry_msgs
    std_msgs
#  DEPENDS system_lib
)