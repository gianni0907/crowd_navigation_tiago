import numpy as np

class Hparams:
    # Specify whether to save data for plots and .json filename
    log = True
    logfile = 'test.json'

    # Kinematic parameters
    wheel_radius = 0.0985 # [m]
    wheel_separation = 0.4044 # [m]
    b = 0.0 # [m]

    # NMPC parameters
    controller_frequency = 50.0 # [Hz]
    N_horizon = 50
    
    # Velocity bounds reduction in case of real_robot
    bound_factor = 1

    # Driving and steering velocity limits
    driving_vel_max = 1 * bound_factor # [m/s]
    driving_vel_min = -0.2 # [m/s]
    steering_vel_max = 1.05 * bound_factor # [rad/s]
    steering_vel_max_neg = -steering_vel_max
    
    # Input velocity limit for both left and right wheel
    w_max = driving_vel_max/wheel_radius # [rad/s]
    w_max_neg = -w_max

    # Configuration limits (only for cartesian position)
    x_lower_bound = -5 # [m]
    x_upper_bound = 5  # [m]
    y_lower_bound = x_lower_bound
    y_upper_bound = x_upper_bound
    
    # State indices:
    x_idx = 0
    y_idx = 1
    theta_idx = 2
    
    # Control input indices
    wr_idx = 0
    wl_idx = 1

    # Tolerance on the position error
    error_tol = 0.05

    # Cost function weights
    q = 1e1 # position weights
    r = 1e-2 # control input weights
    q_factor = 1e1 # factor for the terminal position weights

    # Parameters for the CBF
    rho_cbf = 0.6 # the radius of the circle around the robot center
    ds_cbf = 0.5 # safety clearance
    gamma_cbf = 0.8 # in (0,1], hyperparameter for cbf constraint

    n_obstacles = 5 # number of obstacles

    obstacles_position = np.array([[2.0, 2.0],
                            [2.0, -0.5],
                            [-1.0, 2.3],
                            [-2.0, -1.0],
                            [4.5, 1.0]]) # just to avoid errors in nmpc, it is not useful, will be deleted