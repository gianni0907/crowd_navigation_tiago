import numpy as np

class Hparams:    
    # Specify whether to use real robot or Gazebo
    real_robot = False

    # Specify whether to log /odom and /cmd_vel data in bagfiles
    log = False

    # Kinematic parameters
    wheel_radius = 0.0985 # [m]
    wheel_separation = 0.4044 # [m]
    b = 0.0 # [m]
    
    # Driving and steering velocity limits
    driving_vel_max = 1 # [m/s]
    driving_vel_min = -0.2 # [m/s]
    steering_vel_max = 1.05 # [rad/s]
    steering_vel_max_neg = -steering_vel_max
    
    # Input velocity limit for both left and right wheel
    w_max = driving_vel_max/wheel_radius # [rad/s]
    w_max_neg = -w_max

    # Configuration limits (only for cartesian position)
    lower_bound = -10 # [m]
    upper_bound = 10  # [m]
    
    # State indices:
    x_idx = 0
    y_idx = 1
    theta_idx = 2
    
    # Control input indices
    wr_idx = 0
    wl_idx = 1

