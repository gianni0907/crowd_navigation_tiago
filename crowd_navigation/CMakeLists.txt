cmake_minimum_required(VERSION 3.0.2)
project(crowd_navigation)
find_package(catkin REQUIRED
  crowd_navigation_core
  crowd_nagigation_msgs
  labrob_gazebo_worlds
  labrob_tiago_gazebo
)
catkin_metapackage()