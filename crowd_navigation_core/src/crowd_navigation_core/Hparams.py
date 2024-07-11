import numpy as np
from crowd_navigation_core.utils import *

class Hparams:
    # Specify whether to save data for plots and .json filename
    log = True
    save_video = True
    world_type = WorldType.TWO_ROOMS
    if log:
        log_dir = '/tmp/crowd_navigation_tiago/data'
        filename = 'test'
        generator_file = filename + '_generator.json'
        predictor_file = filename + '_predictor.json'
        laser_file = filename + '_laser.json'
        camera_file = filename + '_camera.json'

    # Specify whether to use gazebo (simulation = True) or real robot
    simulation = True

    # Specify the frequency of the sensors' modules
    if simulation:
        laser_frequency = 10 # [Hz]
        camera_frequency = 15 # [Hz]
    else:
        laser_frequency = 15 # [Hz]
        camera_frequency = 15 # TBD [Hz]

    # Specify the type of sensing, 4 possibilities:
    # GTRUTH: no sensors, the robot knows the ground truth agents' position
    # LASER: only laser sensor enabled
    # CAMERA: only camera enabled
    # BOTH: both laser and camera enabled
    perception = Perception.BOTH

    if perception == Perception.GTRUTH and not simulation:
        raise ValueError("Cannot use ground truth in real world")

    # Specify whether to process measurement with KFs
    use_kalman = True

    # Kinematic parameters
    base_radius = 0.27 # [m]
    wheel_radius = 0.0985 # [m]
    wheel_separation = 0.4044 # [m]
    b = 0.1 # [m]

    # NMPC parameters
    if perception in (Perception.BOTH, Perception.CAMERA):
        generator_frequency = camera_frequency
    elif perception == Perception.LASER:
        generator_frequency = laser_frequency
    elif perception == Perception.GTRUTH: 
        generator_frequency = 15
    T_horizon = 2.7 # [s]
    N_horizon = int(generator_frequency * T_horizon)
    predictor_frequency = generator_frequency
    dt = 1.0 / generator_frequency # [s]
    unbounded = 100000

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
    driving_vel_max = 0.6 * driving_bound_factor # [m/s]
    driving_vel_min = - 0.2 # [m/s]
    steering_vel_max = 1.05 * steering_bound_factor # [rad/s]
    steering_vel_max_neg = - steering_vel_max
    
    # Wheels velocity limits
    w_max = driving_vel_max / wheel_radius # 10.1523 [rad/s]
    w_max_neg = - w_max

    # Define the navigable areas
    ### NOTE: define the points in a counter-clockwise order
    max_vertexes = 6
    if world_type == WorldType.EMPTY:
        if simulation:
            area0 = np.array([[-10, 10], [-10, -10], [10, -10], [10, 10]])
        else:
            area0 = np.array([[-0.6, -4], [4.5, -4], [4.5, 1.8], [-0.6, 1.8]])
        areas = [area0]
        walls = []
    elif world_type == WorldType.TWO_ROOMS:
        area0 = np.array([[4.8, -4.8], [4.8, 4.8], [-4.8, 4.8], [-4.8, -4.8]])
        area1 = np.array([[1, 3.4], [2, 3.4], [2, 6.6], [1, 6.6]])
        area2 = np.array([[4.8, 5.2], [4.8, 9.8], [-4.8, 9.8], [-4.8, 5.2]])
        areas = [area0, area1, area2]
        areas_index = [0, 1, 2]
        intersections = {
            (0, 1), (1, 2)
        }
        viapoints = {
            (0, 1): np.array([1.5, 4.1]),
            (1, 2): np.array([1.5, 5.9])
        }
        walls = [((-5, -5), (-5, 10)),
                ((-5, 10), (5, 10)),
                ((5, 10), (5, -5)),
                ((5, -5), (-5, -5)),
                ((-5, 5), (0.8, 5)),
                ((2.2, 5), (5, 5))]
    elif world_type == WorldType.THREE_ROOMS:
        area0 = np.array([[0, -3.6], [0, -0.2], [-4.8, -0.2], [-4.8, -4.8], [-3,-4.8]])
        area1 = np.array([[4.8, -4.8], [4.8, -2.5], [-3.0, -2.5], [0.0, -4.8]])
        area2 = np.array([[4.8, -4.8], [4.8, -0.2], [1.5, -0.2], [2.5, -4.8]])
        area3 = np.array([[3, -1.6], [3, 1.8], [2, 1.8], [2, -1.6]])
        area4 = np.array([[4.8, 0.4], [4.8, 3.1], [-1.0, 4.8], [-4.8, 4.8], [-4.8, 0.4]])
        area5 = np.array([[-2.0, 3.4], [-2.0, 6.8], [-3.0, 6.8], [-3.0, 3.4]])
        area6 = np.array([[4.8, 5.4], [4.8, 9.8], [-0.5, 9.8], [-4.8, 6.5], [-4.8, 5.4]])
        areas = [area0, area1, area2, area3, area4, area5, area6]
        areas_index = [0, 1, 2, 3, 4, 5, 6]
        intersections = {
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)
        }
        viapoints = {
            (0, 1): np.array([-1.0, -3.0]),
            (1, 2): np.array([3.5, -3.5]),
            (2, 3): np.array([2.5, -0.9]),
            (3, 4): np.array([2.5, 1.1]),
            (4, 5): np.array([-2.5, 4.1]),
            (5, 6): np.array([-2.5, 6.1])
        }
        walls = [((-5, -5), (-5, 10)),
                ((-5, 10), (5, 10)),
                ((5, 10), (5, -5)),
                ((5, -5), (-5, -5)),
                ((-5, 5), (-3.2, 5)),
                ((-1.8, 5), (5, 5)),
                ((3.2, 0), (5, 0)),
                ((-5, 0), (1.8, 0))]
    elif world_type == WorldType.CORRIDOR:
        area0 = np.array([[-2.0, 0.0], [-9.0, 0.0], [-9.0, -2.5], [-4.0, -4.7], [-2.0, -4.7]])
        area1 = np.array([[-8.7, -4.0], [-6.2, -4.0], [-6.2, 4.7], [-7.7, 4.7], [-8.7, 3.7]])
        area2 = np.array([[-7.7, 0.2], [-5.2, 0.2], [-5.2, 9.7], [-6.2, 9.7], [-7.7, 6.0]])
        area3 = np.array([[-4.0, 5.2], [-0.2, 7.0], [-0.2, 8.2], [-3.0, 9.8], [-7.0, 9.8], [-9.8, 5.2]])
        areas = [area0, area1, area2, area3]
        areas_index = [0, 1, 2, 3]
        intersections = {
            (0, 1), (1, 2), (2, 3)
        }
        viapoints = {
            (0, 1): np.array([-7.5, -1.5]),
            (1, 2): np.array([-7.0, 2.5]),
            (2, 3): np.array([-6.0, 7.0])
        }
        walls = [((-10, -5), (0, -5)),
                 ((-10, -5), (-10, 0)),
                 ((0, -5), (0, 0)),
                 ((0, 0), (-6, 0)),
                 ((-10, 0), (-9, 0)),
                 ((-9, 0), (-9, 5)),
                 ((-5, 0), (-5, 5)),
                 ((-8, 5), (-10, 5)),
                 ((-5, 5), (0, 5)),
                 ((0, 5), (0, 10)),
                 ((-10, 5), (-10, 10)),
                 ((0, 10), (-10, 10))]

    a_coefs, b_coefs, c_coefs = get_areas_coefficients(areas, max_vertexes)
    print(get_closest_area_index(areas, np.array([0, -6])))
    
    # State indices:
    x_idx = 0
    y_idx = 1
    theta_idx = 2
    v_idx = 3
    omega_idx = 4
    
    # Control input indices
    r_wheel_idx = 0
    l_wheel_idx = 1

    # Tolerances on the (position and velocity) error
    nmpc_error_tol = 0.15
    pointing_error_tol = 0.4

    # Cost function weights
    if simulation:
        p_weight = 1e2 # position weights
        v_weight = 5e1 # driving velocity weight
        omega_weight = 1e-5 # steering velocity weight
        u_weight = 1e1 # input weights
        h_weight = 120 # heading term weight
        terminal_factor_p = 8e0 # factor for the terminal position weights
        terminal_factor_v = 3e2 # factor for the terminal velocities (v and omega) weights
    else:
        p_weight = 1e2 # position weights
        v_weight = 5e1 # driving velocity weight
        omega_weight = 1e-5 # steering velocity weight
        u_weight = 1e1 # input weights
        h_weight = 120 # heading term weight
        terminal_factor_p = 8e0 # factor for the terminal position weights
        terminal_factor_v = 3e2 # factor for the terminal velocities (v and omega) weights

    # Parameters for the CBF
    rho_cbf = base_radius + b + 0.01 # the radius of the circle around the robot center
    ds_cbf = 0.5 # safety clearance
    gamma_agent = 0.5 # in (0,1], hyperparameter for the h function associated to agent
    gamma_area = 0.1 # in (0,1], hyperparameter for the h function associated to bounds
    
    n_filters = 5 # maximum number of simultaneously tracked agents
    if simulation:
        n_agents = 5 # number of total agents involved, for plotting purpose
        if perception == Perception.GTRUTH:
            n_filters = n_agents

    # Parameters for the crowd prediction
    if n_filters > 0:
        nullpos = -30
        nullstate = np.array([nullpos, nullpos, 0.0, 0.0])
        innovation_threshold = 1
        max_pred_time = dt * N_horizon
        init_cov = 1
        proc_noise_static = 1e-2
        proc_noise_dyn = 1
        meas_noise = 10
        speed_threshold = 0.1
        if simulation:
            offset = 20
        else:
            offset = 10
        # Clustering hparams
        selection_mode = SelectionMode.AVERAGE
        if selection_mode == SelectionMode.CLOSEST:
            eps = 0.8
            min_samples = 2
            avg_win_size = 5
        elif selection_mode == SelectionMode.AVERAGE:
            eps = 0.8
            min_samples = 2

    # Camera Hparams
    if perception in (Perception.BOTH, Perception.CAMERA):
        if simulation:
            cam_min_range = 0.3 # [m]
            cam_max_range = 8 # [m]
            cam_horz_fov = 1.0996 # 1.7453 [rad]
        else:
            cam_min_range = 0.4 # [m]
            cam_max_range = 8 # [m]
            cam_horz_fov = 1.0472 # [rad]
