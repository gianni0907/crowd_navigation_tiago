import numpy as np
import rospy
import math
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Wedge
from  matplotlib.animation import FuncAnimation

from crowd_navigation_core.utils import *
from crowd_navigation_core.Hparams import *

def plot_results(filename=None):
    save_video = Hparams.save_video
    # Specify logging directory
    log_dir = '/tmp/crowd_navigation_tiago/data'
    if not os.path.exists(log_dir):
        raise Exception(
           f"Specified directory not found"
        )

    # Specify saving plots directory
    plots_savedir = '/tmp/crowd_navigation_tiago/plots'
    if not os.path.exists(plots_savedir):
        os.makedirs(plots_savedir)

    # Specify saving animations directory
    animation_savedir = '/tmp/crowd_navigation_tiago/animations'
    if not os.path.exists(animation_savedir):
        os.makedirs(animation_savedir)

    log_generator = os.path.join(log_dir, filename + '_generator.json')
    log_predictor = os.path.join(log_dir, filename + '_predictor.json')
    configuration_savepath = os.path.join(plots_savedir, filename + '_configuration.png')
    velocity_savepath = os.path.join(plots_savedir, filename + '_velocities.png')
    acceleration_savepath = os.path.join(plots_savedir, filename + '_accelerations.png')
    time_savepath = os.path.join(plots_savedir, filename + '_time.png')
    kalman_savepath = os.path.join(plots_savedir, filename + '_kalman.png')
    scans_savepath = os.path.join(animation_savedir, filename + '_scans.mp4')
    world_savepath = os.path.join(animation_savedir, filename + '_world.mp4')

    # Open the generator log file
    if os.path.exists(log_generator):
        with open(log_generator, 'r') as file:
            generator_dict = json.load(file)
    else:
        raise Exception(
            f"Specified file not found"
        )

    # Extract the generator data
    iteration_time = np.array(generator_dict['cpu_time'])
    states = np.array(generator_dict['states'])
    configurations = states[:, :3]
    robot_center = np.empty((configurations.shape[0], 2))
    b = generator_dict['offset_b']
    for i in range(configurations.shape[0]):
        robot_center[i, 0] = configurations[i, 0] - b * math.cos(configurations[i, 2])
        robot_center[i, 1] = configurations[i, 1] - b * math.sin(configurations[i, 2])
    
    driving_velocities = states[:, 3]
    steering_velocities = states[:, 4]
    robot_predictions = np.array(generator_dict['robot_predictions'])
    inputs = np.array(generator_dict['wheels_accelerations'])
    wheels_velocities = np.array(generator_dict['wheels_velocities'])
    commanded_vel = np.array(generator_dict['commanded_velocities'])
    targets = np.array(generator_dict['targets'])
    errors = targets[:, :2] - configurations[:, :2]
    wheel_radius = generator_dict['wheel_radius']
    wheel_separation = generator_dict['wheel_separation']
    driving_acc = wheel_radius * 0.5 * (inputs[:, 0] + inputs[:, 1])
    steering_acc = (wheel_radius / wheel_separation) * (inputs[:, 0] - inputs[:, 1])

    n_edges = generator_dict['n_edges']
    boundary_vertexes = np.array(generator_dict['boundary_vertexes'])
    input_bounds = np.array(generator_dict['input_bounds'])
    v_bounds = np.array(generator_dict['v_bounds'])
    omega_bounds = np.array(generator_dict['omega_bounds'])
    wheels_vel_bounds = np.array(generator_dict['wheels_vel_bounds'])
    vdot_bounds = np.array(generator_dict['vdot_bounds'])
    omegadot_bounds = np.array(generator_dict['omegadot_bounds'])

    n_actors = generator_dict['n_actors']
    n_clusters = generator_dict['n_clusters']
    simulation = generator_dict['simulation']
    if n_actors > 0:
        fake_sensing = generator_dict['fake_sensing']
        use_kalman = generator_dict['use_kalman']
        actors_predictions = np.array(generator_dict['actors_predictions'])
        if simulation and not fake_sensing:
            actors_groundtruth = np.array(generator_dict['actors_gt'])
        
    rho_cbf = generator_dict['rho_cbf']
    ds_cbf = generator_dict['ds_cbf']
    frequency = generator_dict['frequency']
    base_radius = generator_dict['base_radius']
    shooting_nodes = inputs.shape[0]
    t = inputs[:, 2]

    # If the prediction module is present, open the predictor log file
    if n_actors > 0:
        if os.path.exists(log_predictor):
            with open(log_predictor, 'r') as file:
                predictor_dict = json.load(file)
        else:
            raise Exception(
                f"Specified file not found"
            )
        # Extract the predictor data
        kfs_info = predictor_dict['kfs']
        predictor_time = np.array(predictor_dict['cpu_time'])
        robot_states = np.array(predictor_dict['robot_states'])
        robot_config = robot_states[:, :3]
        core_points = predictor_dict['core_points']
        core_points_predictions = np.array(predictor_dict['core_points_predictions'])
        if use_kalman:
            fsm_estimates = np.array(predictor_dict['fsm_estimates'])
        if not fake_sensing:
            laser_scans = predictor_dict['laser_scans']
            angle_inc = predictor_dict['angle_inc']
            offset = predictor_dict['laser_offset']
            angle_min = predictor_dict['angle_min'] + angle_inc * offset
            angle_max = predictor_dict['angle_max'] - angle_inc * offset
            range_min = predictor_dict['range_min']
            range_max = predictor_dict['range_max']
            laser_position = np.array(predictor_dict['laser_relative_pos'])       

    # Figure elapsed time per iteration (generator and predictor if prediction module is present)
    fig, axs = plt.subplots(2, 1, figsize=(16, 8))
    
    axs[0].step(iteration_time[:, 1], iteration_time[:, 0])
    axs[0].set_title('Elapsed time per generator iteration')
    axs[0].set_xlabel('$t \quad [s]$')
    axs[0].set_ylabel('$iteration \quad time \quad [s]$')
    axs[0].hlines(1 / frequency, iteration_time[:, 1], iteration_time[-1, 1], color='red', linestyle='--')
    axs[0].set_xlim([iteration_time[0, 1], iteration_time[-1, 1]])
    axs[0].grid(True)

    if n_actors > 0:
        axs[1].step(predictor_time[:, 1], predictor_time[:, 0])
        axs[1].set_title('Elapsed time per predictor iteration')
        axs[1].set_xlabel('$t \quad [s]$')
        axs[1].set_ylabel('$iteration \quad time \quad [s]$')
        axs[1].hlines(1 / frequency, predictor_time[0, 1], predictor_time[-1, 1], color='red', linestyle='--')
        axs[1].set_xlim([predictor_time[0, 1], predictor_time[-1, 1]])
        axs[1].grid(True)

    fig.tight_layout()
    fig.savefig(time_savepath)

    # Configuration figure
    config_fig, config_ax = plt.subplots(4, 1, figsize=(16, 8))

    config_ax[0].plot(t, configurations[:, 0], label='$x$')
    config_ax[0].plot(t, targets[:, 0], label='$x_g$')
    config_ax[1].plot(t, configurations[:, 1], label='$y$')
    config_ax[1].plot(t, targets[:, 1], label="$y_g$")
    config_ax[2].plot(t, errors[:, 0], label='$e_x$')
    config_ax[2].plot(t, errors[:, 1], label='$e_y$')
    config_ax[3].plot(t, configurations[:, 2], label='$\theta$')

    config_ax[0].set_title('$x-position$')
    config_ax[0].set_xlabel('$t \quad [s]$')
    config_ax[0].set_ylabel('$[m]$')
    config_ax[0].legend(loc='upper left')
    config_ax[0].set_xlim([t[0], t[-1]])
    config_ax[0].grid(True)

    config_ax[1].set_title('$y-position$')
    config_ax[1].set_xlabel('$t \quad [s]$')
    config_ax[1].set_ylabel('$[m]$')
    config_ax[1].legend(loc='upper left')
    config_ax[1].set_xlim([t[0], t[-1]])
    config_ax[1].grid(True)

    config_ax[2].set_title('position errors')
    config_ax[2].set_xlabel("$t \quad [s]$")
    config_ax[2].set_ylabel('$[m]$')
    config_ax[2].legend(loc='upper left')
    config_ax[2].set_xlim([t[0], t[-1]])
    config_ax[2].grid(True)

    config_ax[3].set_title('TIAGo orientation')
    config_ax[3].set_xlabel('$t \quad [s]$')
    config_ax[3].set_ylabel('$[rad]$')
    config_ax[3].set_ylim([-1 + np.min(configurations[:, 2]), 1 + np.max(configurations[:, 2])])
    config_ax[3].set_xlim([t[0], t[-1]])
    config_ax[3].grid(True)

    config_fig.tight_layout()
    config_fig.savefig(configuration_savepath)

    # Velocities figure
    vel_fig, vel_ax = plt.subplots(3, 1, figsize=(16, 8))

    vel_ax[0].plot(t, wheels_velocities[:, 0], label='$\omega^R$')
    vel_ax[0].plot(t, wheels_velocities[:, 1], label='$\omega^L$')
    vel_ax[1].plot(t, driving_velocities, label='$v$')
    vel_ax[1].plot(t, commanded_vel[:, 0], label='$v^{cmd}$')
    vel_ax[2].plot(t, steering_velocities, label='$\omega$')
    vel_ax[2].plot(t, commanded_vel[:, 1], label='$\omega^{cmd}$')

    vel_ax[0].set_title('wheels velocities')
    vel_ax[0].set_xlabel("$t \quad [s]$")
    vel_ax[0].set_ylabel('$[rad/s]$')
    vel_ax[0].legend(loc='upper left')
    vel_ax[0].hlines(wheels_vel_bounds[0], t[0], t[-1], color='red', linestyle='--')
    vel_ax[0].hlines(wheels_vel_bounds[1], t[0], t[-1], color='red', linestyle="--")
    vel_ax[0].set_ylim([-1 + wheels_vel_bounds[0], 1 + wheels_vel_bounds[1]])
    vel_ax[0].set_xlim([t[0], t[-1]])
    vel_ax[0].grid(True)

    vel_ax[1].set_title('TIAGo driving velocity')
    vel_ax[1].set_xlabel("$t \quad [s]$")
    vel_ax[1].set_ylabel('$[m/s]$')
    vel_ax[1].legend(loc='upper left')
    vel_ax[1].hlines(v_bounds[0], t[0], t[-1], color='red', linestyle='--')
    vel_ax[1].hlines(v_bounds[1], t[0], t[-1], color='red', linestyle="--")
    vel_ax[1].set_ylim([-1 + v_bounds[0], 1 + v_bounds[1]])
    vel_ax[1].set_xlim([t[0], t[-1]])
    vel_ax[1].grid(True)

    vel_ax[2].set_title('TIAGo steering velocity')
    vel_ax[2].set_xlabel("$t \quad [s]$")
    vel_ax[2].set_ylabel('$[rad/s]$')
    vel_ax[2].legend(loc='upper left')
    vel_ax[2].hlines(omega_bounds[0], t[0], t[-1], color='red', linestyle='--')
    vel_ax[2].hlines(omega_bounds[1], t[0], t[-1], color='red', linestyle="--")
    vel_ax[2].set_ylim([-1 + omega_bounds[0], 1 + omega_bounds[1]])
    vel_ax[2].set_xlim([t[0], t[-1]])
    vel_ax[2].grid(True)

    vel_fig.tight_layout()
    vel_fig.savefig(velocity_savepath)

    # Accelerations figure
    acc_fig, acc_ax = plt.subplots(3, 1, figsize=(16, 8))

    acc_ax[0].plot(t, inputs[:, 0], label='$\dot{\omega}^R$')
    acc_ax[0].plot(t, inputs[:, 1], label='$\dot{\omega}^L$')
    acc_ax[1].plot(t, driving_acc, label='$\dot{v}$')
    acc_ax[2].plot(t, steering_acc, label='$\dot{omega}$')
    
    acc_ax[0].set_title('wheels accelerations')
    acc_ax[0].set_xlabel("$t \quad [s]$")
    acc_ax[0].set_ylabel('$[rad/s^2]$')
    acc_ax[0].legend(loc='upper left')
    acc_ax[0].hlines(input_bounds[0], t[0], t[-1], color='red', linestyle='--')
    acc_ax[0].hlines(input_bounds[1], t[0], t[-1], color='red', linestyle="--")
    acc_ax[0].set_ylim([-1 + input_bounds[0], 1 + input_bounds[1]])
    acc_ax[0].set_xlim([t[0], t[-1]])
    acc_ax[0].grid(True)

    acc_ax[1].set_title('TIAGo driving acceleration')
    acc_ax[1].set_xlabel("$t \quad [s]$")
    acc_ax[1].set_ylabel('$\dot{v} \ [m/s^2]$')
    acc_ax[1].hlines(vdot_bounds[0], t[0], t[-1], color='red', linestyle='--')
    acc_ax[1].hlines(vdot_bounds[1], t[0], t[-1], color='red', linestyle="--")
    acc_ax[1].set_ylim([-1 + vdot_bounds[0], 1 + vdot_bounds[1]])
    acc_ax[1].set_xlim([t[0], t[-1]])
    acc_ax[1].grid(True)

    acc_ax[2].set_title('TIAGo steering acceleration')
    acc_ax[2].set_xlabel("$t \quad [s]$")
    acc_ax[2].set_ylabel('$\dot{\omega} \ [rad/s^2]$')
    acc_ax[2].hlines(omegadot_bounds[0], t[0], t[-1], color='red', linestyle='--')
    acc_ax[2].hlines(omegadot_bounds[1], t[0], t[-1], color='red', linestyle="--")
    acc_ax[2].set_ylim([-1 + omegadot_bounds[0], 1 + omegadot_bounds[1]])
    acc_ax[2].set_xlim([t[0], t[-1]])
    acc_ax[2].grid(True) 

    acc_fig.tight_layout()
    acc_fig.savefig(acceleration_savepath)

    # Kalman filters figure
    if n_actors > 0 and use_kalman:
        n_kfs = fsm_estimates.shape[1]
        t_predictor = predictor_time[:, 1]
        colors = ['b', 'orange', 'g', 'r', 'c', 'purple', 'brown']
        kfs_fig, kfs_ax = plt.subplots(3, n_kfs, figsize=(16, 8))
        for i in range(n_kfs):
            plot = True
            active = False
            plot_start_idx = 0
            active_start_idx = 0
            distances = np.linalg.norm(robot_config[:, :2] - fsm_estimates[:, i, :2], axis=1)
            for j, time in enumerate(t_predictor):
                if kfs_info[f'KF_{i + 1}'][j][0] == 'FSMStates.ACTIVE':
                    if not active:
                        active = True
                        active_start_idx = j
                else:
                    if active:
                        active = False
                        kfs_ax[0, i].fill_between(t_predictor, 0, 14,
                                                  where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[j]),
                                                  color='gray', alpha='0.1')
                        kfs_ax[1, i].fill_between(t_predictor, -1.4, 1.4,
                                                  where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[j]),
                                                  color='gray', alpha='0.1')
                        kfs_ax[2, i].fill_between(t_predictor, -1.4, 1.4,
                                                  where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[j]),
                                                  color='gray', alpha='0.1')
                
                if all(pos == Hparams.nullpos for pos in fsm_estimates[j, i, :2]):
                    if not plot:
                        kfs_ax[0, i].plot(t_predictor[plot_start_idx : j], distances[plot_start_idx : j], color=colors[i], label='$\hat{d}$')
                        kfs_ax[1, i].plot(t_predictor[plot_start_idx : j], fsm_estimates[plot_start_idx : j, i, 2], color=colors[i], label='$\hat{\dot{p}}_x$')
                        kfs_ax[2, i].plot(t_predictor[plot_start_idx : j], fsm_estimates[plot_start_idx : j, i, 3], color=colors[i], label='$\hat{\dot{p}}_y$')          
                        plot = True
                else:
                    if plot:
                        plot_start_idx = j
                        plot = False

            if active:
                kfs_ax[0, i].fill_between(t_predictor, 0, 14,
                                          where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[-1]),
                                          color='gray', alpha='0.1')
                kfs_ax[1, i].fill_between(t_predictor, -1.4, 1.4,
                                          where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[-1]),
                                          color='gray', alpha='0.1')
                kfs_ax[2, i].fill_between(t_predictor, -1.4, 1.4,
                                          where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[-1]),
                                          color='gray', alpha='0.1')
            if not plot:
                kfs_ax[0, i].plot(t_predictor[plot_start_idx:], distances[plot_start_idx:], color=colors[i], label='$\hat{d}$')
                kfs_ax[1, i].plot(t_predictor[plot_start_idx:], fsm_estimates[plot_start_idx:, i, 2], color=colors[i], label='$\hat{\dot{p}}_x$')
                kfs_ax[2, i].plot(t_predictor[plot_start_idx:], fsm_estimates[plot_start_idx:, i, 3], color=colors[i], label='$\hat{\dot{p}}_y$')

            kfs_ax[0, i].set_title(f'KF-{i + 1}')
            kfs_ax[0, i].set_xlabel("$t \quad [s]$")
            kfs_ax[0, i].set_ylabel('$\hat{d} \quad [m]$')
            kfs_ax[0, i].hlines(rho_cbf + ds_cbf, t_predictor[0], t_predictor[-1], color='red', linestyle='--')
            kfs_ax[1, i].set_xlabel("$t \quad [s]$")
            kfs_ax[1, i].set_ylabel('$\hat{\dot{p}}_x \quad [m/s]$')
            kfs_ax[2, i].set_xlabel("$t \quad [s]$")
            kfs_ax[2, i].set_ylabel('$\hat{\dot{p}}_y \quad [m/s]$')
            kfs_ax[0, i].set_xlim([t_predictor[0], t_predictor[-1]])
            kfs_ax[1, i].set_xlim([t_predictor[0], t_predictor[-1]])
            kfs_ax[2, i].set_xlim([t_predictor[0], t_predictor[-1]])
            kfs_ax[0, i].set_ylim([0, 14])
            kfs_ax[1, i].set_ylim([-1.4, 1.4])
            kfs_ax[2, i].set_ylim([-1.4, 1.4])
            kfs_ax[0, i].grid(True)
            kfs_ax[1, i].grid(True) 
            kfs_ax[2, i].grid(True)

        kfs_fig.tight_layout()
        kfs_fig.savefig(kalman_savepath)

    plt.show()

    # # Figure to plot world animation
    # world_fig = plt.figure(figsize=(16, 8))
    # gs = gridspec.GridSpec(3,2)
    # ax_big = plt.subplot(gs[:, 0])
    # ax1 = plt.subplot(gs[0, 1])
    # ax2 = plt.subplot(gs[1, 1])
    # ax3 = plt.subplot(gs[2, 1])

    # robot = Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='k', label='TIAGo')
    # controlled_pt = ax_big.scatter([], [], marker='.', color='k')
    # robot_label = ax_big.text(np.nan, np.nan, robot.get_label(), fontsize=8, ha='left', va='bottom')
    # robot_clearance = Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='r')
    # goal = ax_big.scatter([], [], s=80.0, marker='*', label='goal', color='magenta', alpha=0.7)
    # goal_label = ax_big.text(np.nan, np.nan, goal.get_label(), fontsize=8, ha='left', va='bottom')
    # if n_actors > 0:
    #     if not fake_sensing:
    #         fov = Wedge(np.zeros(1), np.zeros(1), 0.0, 0.0, color='cyan', alpha=0.1)
    #         if simulation:
    #             actors_gt = []
    #             actors_gt_label = []
    #             actors_gt_clearance = []
    #             for i in range(n_actors):
    #                 actors_gt.append(ax_big.scatter([], [], marker='.', label='actor{}'.format(i+1), color='k', alpha=0.4))
    #                 actors_gt_clearance.append(Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='k', alpha=0.4))
    #                 actors_gt_label.append(ax_big.text(np.nan, np.nan, actors_gt[i].get_label(), fontsize=8, ha='left', va='bottom'))

    #     actors = []
    #     actors_label = []
    #     actors_clearance = []
    #     actors_pred_line = []
    #     for i in range(n_clusters):
    #         actors.append(ax_big.scatter([], [], marker='.', label='fsm{}'.format(i+1), color='red', alpha=0.7))
    #         actors_clearance.append(Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='red'))
    #         actors_label.append(ax_big.text(np.nan, np.nan, actors[i].get_label(), fontsize=8, ha='left', va='bottom'))
    #         actor_pred_line, = ax_big.plot([], [], color='orange', label='actor prediction')
    #         actors_pred_line.append(actor_pred_line)
    
    # traj_line, = ax_big.plot([], [], color='blue', label='trajectory')
    # robot_pred_line, = ax_big.plot([], [], color='green', label='prediction')
    # ex_line, = ax1.plot([], [], label='$e_x$')
    # ey_line, = ax1.plot([], [], label='$e_y$')
    # wr_line, = ax2.plot([], [], label='$\omega_r$')
    # wl_line, = ax2.plot([], [], label='$\omega_l$')
    # alphar_line, = ax3.plot([], [], label='$\\alpha_r$')
    # alphal_line, = ax3.plot([], [], label='$\\alpha_l$')
    # boundary_line = []
    # for i in range(n_edges - 1):
    #     x_values = [boundary_vertexes[i, 0], boundary_vertexes [i + 1, 0]]
    #     y_values = [boundary_vertexes[i, 1], boundary_vertexes [i + 1, 1]]
    #     line, = ax_big.plot(x_values, y_values, color='red', linestyle='--')
    #     boundary_line.append(line)
    # x_values = [boundary_vertexes[n_edges - 1, 0], boundary_vertexes [0, 0]]
    # y_values = [boundary_vertexes[n_edges - 1, 1], boundary_vertexes [0, 1]]
    # line, = ax_big.plot(x_values, y_values, color='red', linestyle='--')
    # boundary_line.append(line)

    # ax_big.set_title('TIAGo World')
    # ax_big.set_xlabel("$x \quad [m]$")
    # ax_big.set_ylabel('$y \quad [m]$')
    # ax_big.set_aspect('equal', adjustable='box')
    # ax_big.grid(True)

    # ax1.set_title('position errors')
    # ax1.set_xlabel("$t \quad [s]$")
    # ax1.set_ylabel('$[m]$')
    # ax1.legend(loc='upper left')
    # ax1.set_ylim([-1 + np.min(errors), 1 + np.max(errors)])
    # ax1.set_xlim([t[0], t[-1]])
    # ax1.grid(True)

    # ax2.set_title('wheels velocities')
    # ax2.set_xlabel("$t \quad [s]$")
    # ax2.set_ylabel('$[rad/s]$')
    # ax2.legend(loc='upper left')
    # ax2.hlines(wheels_vel_bounds[0], t[0], t[-1], color='red', linestyle='--')
    # ax2.hlines(wheels_vel_bounds[1], t[0], t[-1], color='red', linestyle="--")
    # ax2.set_ylim([-1 + wheels_vel_bounds[0], 1 + wheels_vel_bounds[1]])
    # ax2.set_xlim([t[0], t[-1]])
    # ax2.grid(True)

    # ax3.set_title('wheels accelerations')
    # ax3.set_xlabel("$t \quad [s]$")
    # ax3.set_ylabel('$[rad/s^2]$')
    # ax3.legend(loc='upper left')
    # ax3.hlines(input_bounds[0], t[0], t[-1], color='red', linestyle='--')
    # ax3.hlines(input_bounds[1], t[0], t[-1], color='red', linestyle="--")
    # ax3.set_ylim([-1 + input_bounds[0], 1 + input_bounds[1]])
    # ax3.set_xlim([t[0], t[-1]])
    # ax3.grid(True)

    # # init and update function for the world animation
    # def init_world():
    #     robot.set_center(robot_center[0])
    #     robot.set_radius(base_radius)
    #     ax_big.add_patch(robot)
    #     controlled_pt.set_offsets(configurations[0, :2])
    #     robot_clearance.set_center(configurations[0, :2])
    #     robot_clearance.set_radius(rho_cbf)
    #     ax_big.add_patch(robot_clearance)
    #     robot_label.set_position(robot_center[0])

    #     goal.set_offsets(targets[0, :2])
    #     goal_label.set_position(targets[0])

    #     if n_actors > 0:        
    #         for i in range(n_clusters):
    #             actor_position = actors_predictions[0, i, :, 0]
    #             actors[i].set_offsets(actor_position)
    #             actors_clearance[i].set_center(actor_position)
    #             actors_clearance[i].set_radius(ds_cbf)
    #             ax_big.add_patch(actors_clearance[i])
    #             actors_label[i].set_position(actor_position)

    #         if simulation and not fake_sensing:
    #             ax_big.add_patch(fov)
    #             for i in range(n_actors):
    #                 actor_gt_position = actors_groundtruth[0, i, :]
    #                 actors_gt[i].set_offsets(actor_gt_position)
    #                 actors_gt_clearance[i].set_center(actor_gt_position)
    #                 actors_gt_clearance[i].set_radius(ds_cbf)
    #                 ax_big.add_patch(actors_gt_clearance[i])
    #                 actors_gt_label[i].set_position(actor_gt_position)

    #             return robot, robot_clearance, robot_label, goal, goal_label, \
    #                    actors, actors_clearance, actors_label, fov, \
    #                    actors_gt, actors_gt_clearance, actors_gt_label
    #         else:
    #             return robot, robot_clearance, robot_label, goal, goal_label, \
    #                    actors, actors_clearance, actors_label
    #     else:
    #         return robot, robot_clearance, robot_label, goal, goal_label

    # def update_world(frame):
    #     if frame == shooting_nodes - 1:
    #         world_animation.event_source.stop()
        
    #     robot_prediction = robot_predictions[frame, :, :]
    #     current_target = targets[frame, :]

    #     ex_line.set_data(t[:frame + 1], targets[:frame + 1, 0] - configurations[:frame + 1, 0])
    #     ey_line.set_data(t[:frame + 1], targets[:frame + 1, 1] - configurations[:frame + 1, 1])
    #     wr_line.set_data(t[:frame + 1], wheels_velocities[:frame + 1, 0])
    #     wl_line.set_data(t[:frame + 1], wheels_velocities[:frame + 1, 1])
    #     alphar_line.set_data(t[:frame + 1], inputs[:frame + 1, 0])
    #     alphal_line.set_data(t[:frame + 1], inputs[:frame + 1, 1])
    #     traj_line.set_data(configurations[:frame + 1, 0], configurations[:frame + 1, 1])
    #     robot_pred_line.set_data(robot_prediction[0, :], robot_prediction[1, :])

    #     robot.set_center(robot_center[frame])
    #     controlled_pt.set_offsets(configurations[frame, :2])
    #     robot_clearance.set_center(configurations[frame, :2])
    #     robot_label.set_position(robot_center[frame])
    #     goal.set_offsets(current_target[:2])
    #     goal_label.set_position(current_target)

    #     if n_actors > 0:
    #         for i in range(n_clusters):
    #             actor_prediction = actors_predictions[frame, i, :, :]
    #             actor_position = actor_prediction[: , 0]
    #             actors[i].set_offsets(actor_position)
    #             actors_clearance[i].set_center(actor_position)
    #             actors_label[i].set_position(actor_position)
    #             actors_pred_line[i].set_data(actor_prediction[0, :], actor_prediction[1, :])
                
    #         if not fake_sensing:
    #             theta = configurations[frame, 2]
    #             current_laser_pos = configurations[frame, :2] + z_rotation(theta, laser_position)
    #             fov.set_center(current_laser_pos)
    #             fov.set_radius(range_max)
    #             fov.set_theta1((theta + angle_min) * 180 / np.pi)
    #             fov.set_theta2((theta + angle_max) * 180 / np.pi)
    #             fov.set_width(range_max - range_min) 

    #             if simulation:
    #                 for i in range(n_actors):
    #                     actor_gt_position = actors_groundtruth[frame, i, :]
    #                     actors_gt[i].set_offsets(actor_gt_position)
    #                     actors_gt_clearance[i].set_center(actor_gt_position)
    #                     actors_gt_label[i].set_position(actor_gt_position)

    #                 return robot, robot_clearance, robot_label, goal, goal_label, \
    #                     ex_line, ey_line, wr_line, wl_line, alphar_line, alphal_line, \
    #                     traj_line, robot_pred_line, fov, \
    #                     actors, actors_clearance, actors_label, actors_pred_line, \
    #                     actors_gt, actors_gt_clearance, actors_gt_label
                
    #             return robot, robot_clearance, robot_label, goal, goal_label, \
    #                     ex_line, ey_line, wr_line, wl_line, alphar_line, alphal_line, \
    #                     traj_line, robot_pred_line, fov, \
    #                     actors, actors_clearance, actors_label, actors_pred_line
    #         else:
    #             return robot, robot_clearance, robot_label, goal, goal_label, \
    #                    ex_line, ey_line, wr_line, wl_line, alphar_line, alphal_line, \
    #                    traj_line, robot_pred_line, \
    #                    actors, actors_clearance, actors_label, actors_pred_line
    #     else:
    #         return robot, robot_clearance, robot_label, goal, goal_label, \
    #                ex_line, ey_line, wr_line, wl_line, alphar_line, alphal_line, \
    #                traj_line, robot_pred_line

    # Figure to plot world animation
    world_fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1,1)
    ax_wrld = plt.subplot(gs[0, 0])

    robot = Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='k', label='TIAGo')
    controlled_pt = ax_wrld.scatter([], [], marker='.', color='k')
    robot_label = ax_wrld.text(np.nan, np.nan, robot.get_label(), fontsize=16, ha='left', va='bottom')
    robot_clearance = Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='r', linestyle='--')
    goal = ax_wrld.scatter([], [], s=100, marker='*', label='goal', color='magenta')
    goal_label = ax_wrld.text(np.nan, np.nan, goal.get_label(), fontsize=16, ha='left', va='bottom')
    if n_actors > 0:
        if not fake_sensing:
            fov = Wedge(np.zeros(1), np.zeros(1), 0.0, 0.0, color='cyan', alpha=0.1)
            if simulation:
                actors_gt = []
                actors_gt_label = []
                actors_gt_clearance = []
                for i in range(n_actors):
                    actors_gt.append(ax_wrld.scatter([], [], marker='.', label='hum{}'.format(i+1), color='k', alpha=0.3))
                    actors_gt_clearance.append(Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='k', linestyle='--', alpha=0.3))
                    actors_gt_label.append(ax_wrld.text(np.nan, np.nan, actors_gt[i].get_label(), fontsize=16, ha='left', va='bottom', alpha=0.3))

        actors = []
        actors_label = []
        actors_clearance = []
        actors_pred_line = []
        for i in range(n_clusters):
            actors.append(ax_wrld.scatter([], [], marker='.', label='KF-{}'.format(i+1), color='red'))
            actors_clearance.append(Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='red', linestyle='--'))
            actors_label.append(ax_wrld.text(np.nan, np.nan, actors[i].get_label(), fontsize=16, ha='left', va='bottom'))
            actor_pred_line, = ax_wrld.plot([], [], color='orange', label='actor prediction')
            actors_pred_line.append(actor_pred_line)

    traj_line, = ax_wrld.plot([], [], color='blue', label='trajectory')
    robot_pred_line, = ax_wrld.plot([], [], color='green', label='prediction')
    boundary_line = []
    for i in range(n_edges - 1):
        x_values = [boundary_vertexes[i, 0], boundary_vertexes [i + 1, 0]]
        y_values = [boundary_vertexes[i, 1], boundary_vertexes [i + 1, 1]]
        line, = ax_wrld.plot(x_values, y_values, color='red', linestyle='--')
        boundary_line.append(line)
    x_values = [boundary_vertexes[n_edges - 1, 0], boundary_vertexes [0, 0]]
    y_values = [boundary_vertexes[n_edges - 1, 1], boundary_vertexes [0, 1]]
    line, = ax_wrld.plot(x_values, y_values, color='red', linestyle='--')
    boundary_line.append(line)

    ax_wrld.set_title('TIAGo World')
    ax_wrld.set_xlabel("$x \quad [m]$")
    ax_wrld.set_ylabel('$y \quad [m]$')
    ax_wrld.set_aspect('equal', adjustable='box')
    ax_wrld.grid(True)

    # init and update function for the world animation
    def init_world():
        robot.set_center(robot_center[0])
        robot.set_radius(base_radius)
        ax_wrld.add_patch(robot)
        controlled_pt.set_offsets(configurations[0, :2])
        robot_clearance.set_center(configurations[0, :2])
        robot_clearance.set_radius(rho_cbf)
        ax_wrld.add_patch(robot_clearance)
        robot_label.set_position(robot_center[0])

        goal.set_offsets(targets[0, :2])
        goal_label.set_position(targets[0])
        if n_actors > 0:        
            for i in range(n_clusters):
                actor_position = actors_predictions[0, i, :, 0]
                actors[i].set_offsets(actor_position)
                actors_clearance[i].set_center(actor_position)
                actors_clearance[i].set_radius(ds_cbf)
                ax_wrld.add_patch(actors_clearance[i])
                actors_label[i].set_position(actor_position)

            if not fake_sensing:
                ax_wrld.add_patch(fov)
            if simulation and not fake_sensing:
                for i in range(n_actors):
                    actor_gt_position = actors_groundtruth[0, i, :]
                    actors_gt[i].set_offsets(actor_gt_position)
                    actors_gt_clearance[i].set_center(actor_gt_position)
                    actors_gt_clearance[i].set_radius(ds_cbf)
                    ax_wrld.add_patch(actors_gt_clearance[i])
                    actors_gt_label[i].set_position(actor_gt_position)

                return robot, robot_clearance, robot_label, goal, goal_label, \
                       actors, actors_clearance, actors_label, fov, \
                       actors_gt, actors_gt_clearance, actors_gt_label
            else:
                return robot, robot_clearance, robot_label, goal, goal_label, \
                       actors, actors_clearance, actors_label
        else:
            return robot, robot_clearance, robot_label, goal, goal_label
        
    def update_world(frame):
        if frame == shooting_nodes - 1:
            world_animation.event_source.stop()

        robot_prediction = robot_predictions[frame, :, :]
        current_target = targets[frame, :]
        traj_line.set_data(configurations[:frame + 1, 0], configurations[:frame + 1, 1])
        robot_pred_line.set_data(robot_prediction[0, :], robot_prediction[1, :])

        robot.set_center(robot_center[frame])
        controlled_pt.set_offsets(configurations[frame, :2])
        robot_clearance.set_center(configurations[frame, :2])
        robot_label.set_position(robot_center[frame])
        goal.set_offsets(current_target[:2])
        goal_label.set_position(current_target)
        if n_actors > 0:
            for i in range(n_clusters):
                actor_prediction = actors_predictions[frame, i, :, :]
                actor_position = actor_prediction[: , 0]
                actors[i].set_offsets(actor_position)
                actors_clearance[i].set_center(actor_position)
                actors_label[i].set_position(actor_position)
                actors_pred_line[i].set_data(actor_prediction[0, :], actor_prediction[1, :])
                
            if not fake_sensing:
                theta = configurations[frame, 2]
                current_laser_pos = configurations[frame, :2] + z_rotation(theta, laser_position)
                fov.set_center(current_laser_pos)
                fov.set_radius(range_max)
                fov.set_theta1((theta + angle_min) * 180 / np.pi)
                fov.set_theta2((theta + angle_max) * 180 / np.pi)
                fov.set_width(range_max - range_min)

                if simulation:
                    for i in range(n_actors):
                        actor_gt_position = actors_groundtruth[frame, i, :]
                        actors_gt[i].set_offsets(actor_gt_position)
                        actors_gt_clearance[i].set_center(actor_gt_position)
                        actors_gt_label[i].set_position(actor_gt_position)

                    return robot, robot_clearance, robot_label, goal, goal_label, \
                        traj_line, robot_pred_line, fov, \
                        actors, actors_clearance, actors_label, actors_pred_line, \
                        actors_gt, actors_gt_clearance, actors_gt_label

                return robot, robot_clearance, robot_label, goal, goal_label, \
                        traj_line, robot_pred_line, fov, \
                        actors, actors_clearance, actors_label, actors_pred_line
            else:
                return robot, robot_clearance, robot_label, goal, goal_label, \
                       traj_line, robot_pred_line, \
                       actors, actors_clearance, actors_label, actors_pred_line
        else:
            return robot, robot_clearance, robot_label, goal, goal_label, \
                   traj_line, robot_pred_line
    
    world_animation = FuncAnimation(world_fig, update_world,
                                    frames=shooting_nodes,
                                    init_func=init_world,
                                    blit=False,
                                    interval=1/frequency*500,
                                    repeat=False)
    world_fig.tight_layout()
    if save_video:
        world_animation.save(world_savepath, writer='ffmpeg', fps=frequency, dpi=80)
        print("World animation saved")
    
    plt.show()

    # Figure to plot scans animation
    if n_actors > 0 and not fake_sensing:
        scans_fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(1,1)
        ax = plt.subplot(gs[0, 0])

        robot = Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='k', label='TIAGo')
        controlled_pt = ax.scatter([], [], marker='.', color='k')
        robot_label = ax.text(np.nan, np.nan, robot.get_label(), fontsize=16, ha='left', va='bottom')
        scans, = ax.plot([], [], color='magenta', marker='.', markersize=3, linestyle='', label='scans')
        fov = Wedge(np.zeros(1), np.zeros(1), 0.0, 0.0, color='cyan', alpha=0.1)

        core_pred_line = []
        core_points_position = []
        core_points_label = []        
        for i in range(n_clusters):
            core_points_position.append(ax.scatter([], [], marker='.', label='KF-${}$'.format(i+1), color='b'))
            core_points_label.append(ax.text(np.nan, np.nan, core_points_position[i].get_label(), fontsize=16, ha='left', va='bottom'))
            pt_pred_line, = ax.plot([], [], color='orange', label='actor prediction')
            core_pred_line.append(pt_pred_line)

        boundary_line = []
        for i in range(n_edges - 1):
            x_values = [boundary_vertexes[i, 0], boundary_vertexes [i + 1, 0]]
            y_values = [boundary_vertexes[i, 1], boundary_vertexes [i + 1, 1]]
            line, = ax.plot(x_values, y_values, color='red', linestyle='--')
            boundary_line.append(line)
        x_values = [boundary_vertexes[n_edges - 1, 0], boundary_vertexes [0, 0]]
        y_values = [boundary_vertexes[n_edges - 1, 1], boundary_vertexes [0, 1]]
        line, = ax.plot(x_values, y_values, color='red', linestyle='--')
        boundary_line.append(line)

        ax.set_title('TIAGo Scans')
        ax.set_xlabel("$x \quad [m]$")
        ax.set_ylabel('$y \quad [m]$')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

        shooting_nodes = robot_config.shape[0]
        robot_center = np.empty((robot_config.shape[0], 2))
        for i in range(robot_config.shape[0]):
            robot_center[i, 0] = robot_config[i, 0] - b * math.cos(robot_config[i, 2])
            robot_center[i, 1] = robot_config[i, 1] - b * math.sin(robot_config[i, 2])

        # init and update function for the scans animation
        def init_scans():
            robot.set_center(robot_center[0])
            robot.set_radius(base_radius)
            ax.add_patch(robot)
            ax.add_patch(fov)
            controlled_pt.set_offsets(robot_config[0, :2])
            robot_label.set_position(robot_center[0])

            for i in range(n_clusters):
                core_point_position = core_points_predictions[0, i, :, 0]
                core_points_position[i].set_offsets(core_point_position)
                core_points_label[i].set_position(core_points_position)
        
            return robot, fov, robot_label, core_points_position, core_points_label
        
        def update_scans(frame):
            if frame == shooting_nodes - 1:
                scans_animation.event_source.stop()

            robot.set_center(robot_center[frame])
            controlled_pt.set_offsets(robot_config[frame, :2])
            robot_label.set_position(robot_center[frame])
            current_scans = np.array(laser_scans[frame])

            theta = robot_config[frame, 2]
            current_laser_pos = robot_config[frame, :2] + z_rotation(theta, laser_position)
            fov.set_center(current_laser_pos)
            fov.set_radius(range_max)
            fov.set_theta1((theta + angle_min) * 180 / np.pi)
            fov.set_theta2((theta + angle_max) * 180 / np.pi)
            fov.set_width(range_max - range_min)

            for i in range(n_clusters):
                core_point_prediction = core_points_predictions[frame, i, :, :]
                core_point_position = core_point_prediction[: , 0]
                core_points_position[i].set_offsets(core_point_position)
                core_points_label[i].set_position(core_point_position)
                core_pred_line[i].set_data(core_point_prediction[0, :], core_point_prediction[1, :])


            if current_scans.shape[0] > 0:
                scans.set_data(current_scans[:, 0], current_scans[:, 1])
            else:
                scans.set_data([], [])

            return robot, robot_label, fov, scans, core_points_position, core_points_label, core_pred_line

        scans_animation = FuncAnimation(scans_fig, update_scans,
                                        frames=shooting_nodes,
                                        init_func=init_scans,
                                        blit=False,
                                        interval=1/frequency*500,
                                        repeat=False)
        scans_fig.tight_layout()
        if save_video:
            scans_animation.save(scans_savepath, writer='ffmpeg', fps=frequency, dpi=80)
            print("Scans animation saved")
        
        plt.show()

def main():
    filename = rospy.get_param('/filename')
    plot_results(filename)