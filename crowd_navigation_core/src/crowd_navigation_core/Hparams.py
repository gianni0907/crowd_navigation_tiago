import numpy as np
from crowd_navigation_core.utils import *

class Hparams:
    # Specify whether to save data for plots and .json filename
    log = True
    save_video = True
    if log:
        log_dir = '/tmp/crowd_navigation_tiago/data'
        filename = 'test_both_camerafirst'
        generator_file = filename + '_generator.json'
        predictor_file = filename + '_predictor.json'
        laser_detector_file = filename + '_laser_detector.json'
        camera_detector_file = filename + '_camera_detector.json'

    # Specify whether to use gazebo (simulation = True) or real robot
    simulation = True

    # Specify the frequency of the sensors' modules
    if simulation:
        laser_detector_frequency = 10 # [Hz]
        camera_detector_frequency = 30 # [Hz]
    else:
        laser_detector_frequency = 15 # [Hz]
        camera_detector_frequency = 30 # TBD [Hz]

    # Specify the type of sensing, 4 possibilities:
    # FAKE: no sensors, the robot knows the fake trajectory assigned to agents (not visible in Gazebo)
    # LASER: only lasser sensor enabled
    # CAMERA: only camera enabled
    # BOTH: both laser and camera enabled
    perception = Perception.BOTH

    if perception == Perception.FAKE and not simulation:
        raise ValueError("Cannot use fake perception in real world")

    # Specify whether to process measurement with KFs
    use_kalman = True

    if perception == Perception.FAKE and use_kalman == True:
        raise ValueError("Cannot use KFs with fake sensing")

    # Kinematic parameters
    base_radius = 0.27 # [m]
    wheel_radius = 0.0985 # [m]
    wheel_separation = 0.4044 # [m]
    b = 0.1 # [m]

    # NMPC parameters
    if perception in (Perception.BOTH, Perception.CAMERA):
        generator_frequency = camera_detector_frequency
        N_horizon = 75
    elif perception == Perception.LASER:
        generator_frequency = laser_detector_frequency
        N_horizon = 25
    else:
        generator_frequency = 20
        N_horizon = 50
    predictor_frequency = generator_frequency
    dt = 1.0 / generator_frequency # [s]
    unbounded = 1000

    # Driving and steering acceleration limits
    driving_acc_max = 1.0 # [m/s^2]
    driving_acc_min = - driving_acc_max
    steering_acc_max = 1.05 # [rad/s^2]
    steering_acc_max_neg = - steering_acc_max

    # Wheels acceleration limits
    alpha_max = driving_acc_max / wheel_radius # 10.1523 [rad/s^2]
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
    w_max = driving_vel_max / wheel_radius # 10.1523 [rad/s]
    w_max_neg = - w_max

    # Set n points to be the vertexes of the admitted region
    ### NOTE: define the points in a counter-clockwise order
    n_points = 4
    if simulation:
        vertexes = np.array([[-6, 6],
                             [-6, -6],
                             [6, -6],
                             [6, 6]])
        # vertexes = np.array([[-1.5, 11.5],
        #                      [-1.5, -1.5],
        #                      [11.5, -1.5],
        #                      [11.5, 11.5]])
    else:
        vertexes = np.array([[-3.9, 0.0],
                             [-3.0, -2.4],
                             [2.6, -0.2],
                             [1.8, 2.0]])
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
    error_tol = 0.1

    # Cost function weights
    if simulation:
        p_weight = 1e2 # position weights
        v_weight = 5e1 # driving velocity weight
        omega_weight = 1e-5 # steering velocity weight
        u_weight = 1e1 # input weights
        h_weight = 1e2 # heading term weight
        terminal_factor_p = 8e0 # factor for the terminal position weights
        terminal_factor_v = 3e2 # factor for the terminal velocities (v and omega) weights
    else:
        p_weight = 1e2 # position weights
        v_weight = 5e1 # driving velocity weight
        omega_weight = 1e-5 # steering velocity weight
        u_weight = 1e1 # input weights
        h_weight = 1e2 # heading term weight
        terminal_factor_p = 8e0 # factor for the terminal position weights
        terminal_factor_v = 3e2 # factor for the terminal velocities (v and omega) weights

    # Parameters for the CBF
    rho_cbf = base_radius + b + 0.01 # the radius of the circle around the robot center
    ds_cbf = 0.4 # safety clearance
    gamma_agent = 0.1 # in (0,1], hyperparameter for the h function associated to agent
    gamma_bound = 0.1 # in (0,1], hyperparameter for the h function associated to bounds
    
    n_filters = 5 # maximum number of simultaneously tracked agents
    if simulation:
        n_agents = 5 # number of total agents involved, for plotting purpose
        if perception == Perception.FAKE:
            n_filters = n_agents

    # Parameters for the crowd prediction
    if n_filters > 0:
        nullpos = -30
        nullstate = np.array([nullpos, nullpos, 0.0, 0.0])
        innovation_threshold = 1
        max_pred_time = dt * N_horizon
        if simulation:
            offset = 20
        else:
            offset = 10
        # Clustering hparams
        selection_mode = SelectionMode.AVERAGE
        if selection_mode == SelectionMode.CLOSEST:
            eps = 0.7
            min_samples = 2
            avg_win_size = 5
        elif selection_mode == SelectionMode.AVERAGE:
            eps = 0.7
            min_samples = 2