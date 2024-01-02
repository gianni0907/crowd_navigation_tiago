import numpy as np

class Hparams:
    # Specify whether to save data for plots and .json filename
    log = True
    controller_file = 'test_controller.json'
    prediction_file = 'test_predictor.json'

    # Kinematic parameters
    wheel_radius = 0.0985 # [m]
    wheel_separation = 0.4044 # [m]
    b = 0.1 # [m]

    # NMPC parameters
    controller_frequency = 40.0 # [Hz]
    dt = 1 / controller_frequency # [s]
    N_horizon = 20

    # Driving and steering acceleration limits
    driving_acc_max = 0.5 # [m/s^2]
    driving_acc_min = - driving_acc_max
    steering_acc_max = 1.05 # [rad/s^2]
    steering_acc_max_neg = - steering_acc_max

    # Wheels acceleration limits
    alpha_max = driving_acc_max / wheel_radius # [rad/s^2], 5.0761
    alpha_min = - alpha_max

    # Velocity bounds reduction in case of real_robot
    driving_bound_factor = 1.0
    steering_bound_factor = 1.0
    
    # Driving and steering velocity limits
    driving_vel_max = 1 * driving_bound_factor # [m/s]
    driving_vel_min = - 0.2 # [m/s]
    steering_vel_max = 1.05 * steering_bound_factor # [rad/s]
    steering_vel_max_neg = - steering_vel_max
    
    # Wheels velocity limits
    w_max = 1.05 * driving_vel_max / wheel_radius # [rad/s], 10.1523
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
    r_wheel_idx = 0
    l_wheel_idx = 1

    # Tolerance on the position error
    error_tol = 0.05

    # Cost function weights
    p_weight = 1e2 # position weights
    v_weight = 1e4 # driving velocity weight
    omega_weight = 1e0 # steering velocity weight
    u_weight = 1e0 # input weights
    terminal_factor = 1e2 # factor for the terminal state

    # Parameters for the CBF
    rho_cbf = 0.4 # the radius of the circle around the robot center
    ds_cbf = 0.5 # safety clearance
    gamma_actor = 0.8 # in (0,1], hyperparameter for cbf constraint associated to actors
    gamma_bound = 0.8 # in (0,1], hyperparameter for cbf constraint associated to bounds
    n_actors = 3 # number of actors

    # Parameters for the crowd prediction
    innovation_threshold = 1
    matching_threshold = 0.1
    max_pred_time = 1
