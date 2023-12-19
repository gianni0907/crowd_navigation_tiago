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
    controller_frequency = 40.0 # [Hz]
    N_horizon = 20

    # Driving and steering acceleration limits
    driving_acc_max = 0.5 # [m/s^2]
    driving_acc_min = - driving_acc_max
    steering_acc_max = 1.05 # [rad/s^2]
    steering_acc_max_neg = - steering_acc_max

    # Wheels acceleration limits
    alpha_max = driving_acc_max / wheel_radius # [rad/s^2], 5,0761
    alpha_min = - alpha_max

    # Driving and steering velocity limits
    driving_vel_max = 1 # [m/s]
    driving_vel_min = -0.2 # [m/s]
    steering_vel_max = 1.05 # [rad/s]
    steering_vel_max_neg = - steering_vel_max
    
    # Wheels velocity limits
    w_max = driving_vel_max / wheel_radius # [rad/s], 10.1523
    w_max_neg = - w_max

    # Configuration limits (only for cartesian position)
    x_lower_bound = -5 # [m]
    x_upper_bound = 5  # [m]
    y_lower_bound = x_lower_bound
    y_upper_bound = x_upper_bound
    
    # State indices:
    x_idx = 0
    y_idx = 1
    theta_idx = 2
    v_idx = 3
    omega_idx = 4
    
    # Control input indices
    alphar_idx = 0
    alphal_idx = 1

    # Tolerance on the position error
    error_tol = 0.05

    # Cost function weights
    p_weight = 1e1 # position weights
    v_weight = 1e-2 # driving velocity weight
    omega_weight = 1e-2 # steering velocity weight
    u_weight = 1e-2 # input weights
    terminal_factor = 1e1 # factor for the terminal state

    # Parameters for the CBF
    rho_cbf = 0.6 # the radius of the circle around the robot center
    ds_cbf = 0.5 # safety clearance
    gamma_cbf = 0.5 # in (0,1], hyperparameter for cbf constraint
    n_obstacles = 0 # number of obstacles, for now static obstacles
    obstacles_position = np.array([[2.0, 2.0],
                                   [2.0, -0.5],
                                   [-1.0, 2.3],
                                   [-2.0, -1.0],
                                   [4.5, 1.0]]) # fixed position of the obstacles (for NMPC, to be deleted)