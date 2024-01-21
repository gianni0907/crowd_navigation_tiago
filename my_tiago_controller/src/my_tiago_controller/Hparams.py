import numpy as np
from my_tiago_controller.utils import *

class Hparams:
    # Specify whether to save data for plots and .json filename
    log = True
    if log:
        filename = 'smoothed'
        controller_file = filename + '_controller.json'
        prediction_file = filename + '_predictor.json'

    # Specify whether to use gazebo or real robot
    simulation = True

    # Specify whether to use laser scans data or ground truth
    fake_sensing = False

    # Specify whether to process measurement with KFs
    use_kalman = True

    # Kinematic parameters
    wheel_radius = 0.0985 # [m]
    wheel_separation = 0.4044 # [m]
    b = 0.1 # [m]
    relative_laser_pos = np.array([0.2012 - b, -0.0009])

    # NMPC parameters
    controller_frequency = 18.0 # [Hz]
    dt = 2.0 / controller_frequency # [s]
    N_horizon = 10

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

    # Set n points to be the vertexes of the admitted region
    n_points = 4
    vertexes = np.array([[-1.0, 0.5],
                         [-1.0, -1.4],
                         [8.0, -1.4],
                         [8.0, 0.5]])

    normals = np.zeros((n_points, 2))
    for i in range(n_points - 1):
        normals[i] = compute_normal_vector(vertexes[i], vertexes[i + 1])
    normals[n_points - 1] = compute_normal_vector(vertexes[n_points - 1], vertexes[0])
    
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
    if simulation:
        p_weight = 1e2 # position weights
        v_weight = 5e0 # driving velocity weight
        omega_weight = 1e-5 # steering velocity weight
        u_weight = 1e1 # input weights
        terminal_factor_p = 1e1 # factor for the terminal position weights
        terminal_factor_v = 8e1 # factor for the terminal velocities (v and omega) weights
    else:
        p_weight = 1e2 # position weights
        v_weight = 5e0 # driving velocity weight
        omega_weight = 1e-5 # steering velocity weight
        u_weight = 1e1 # input weights
        terminal_factor_p = 1e1 # factor for the terminal position weights
        terminal_factor_v = 8e1 # factor for the terminal velocities (v and omega) weights

    # Parameters for the CBF
    rho_cbf = 0.3 # the radius of the circle around the robot center
    ds_cbf = 0.2 # safety clearance
    gamma_actor = 0.1 # in (0,1], hyperparameter for the h function associated to actor
    gamma_bound = 0.3 # in (0,1], hyperparameter for the h function associated to bounds
    
    n_actors = 2 # number of actors
    if n_actors == 0 or fake_sensing:
        n_clusters = n_actors
    else:
        n_clusters = 2 # number of clusters

    # Parameters for the crowd prediction
    if n_actors > 0:
        nullstate = np.array([-30, -30, 0.0, 0.0])
        innovation_threshold = 0.5
        matching_threshold = 0.1
        max_pred_time = dt * N_horizon
