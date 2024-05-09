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

class Plotter:
    '''
    Plot the data and animations about the sensors (camera and laser),
    the crowd prediction and the motion generation modules 
    '''
    def __init__(self, filename):
        self.filename = filename
        self.save_video = Hparams.save_video

        # Specify logging directory
        log_dir = Hparams.log_dir
        if not os.path.exists(log_dir):
            raise Exception(f"Specified directory not found")

        # Set the loggers
        self.log_generator = os.path.join(log_dir, filename + '_generator.json')
        self.log_predictor = os.path.join(log_dir, filename + '_predictor.json')
        self.log_laser_detector = os.path.join(log_dir, filename + '_laser_detector.json')
        self.log_camera_detector = os.path.join(log_dir, filename + '_camera_detector.json')

        # Specify saving plots directory
        self.plots_dir = '/tmp/crowd_navigation_tiago/plots'
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

        # Specify saving animations directory
        self.animation_dir = '/tmp/crowd_navigation_tiago/animations'
        if not os.path.exists(self.animation_dir):
            os.makedirs(self.animation_dir)

    def plot_times(self):
        # Specify the saving path
        time_path = os.path.join(self.plots_dir, self.filename + '_time.png')

        # Open the generator log file
        if os.path.exists(self.log_generator):
            with open(self.log_generator, 'r') as file:
                generator_dict = json.load(file)
            # Extract the generator time
            generator_time = np.array(generator_dict['cpu_time'])
        else:
            raise Exception(
                f"Generator logfile not found"
            )

        # Open the predictor log file
        if os.path.exists(self.log_predictor):
            with open(self.log_predictor, 'r') as file:
                predictor_dict = json.load(file)
            # Extract the predictor time
            predictor_time = np.array(predictor_dict['cpu_time'])
        else:
            raise Exception(
                f"Predictor logfile not found"
            )
        
        # Open the camera detector log file
        if os.path.exists(self.log_camera_detector):
            with open(self.log_camera_detector, 'r') as file:
                camera_detector_dict = json.load(file)
            # Extract the camera detector time
            camera_detector_time = np.array(camera_detector_dict['cpu_time'])
            camera_frequency = camera_detector_dict['frequency']
        else:
            raise Exception(
                f"Camera detector logfile not found"
            )

        # Open the laser detector log file
        if os.path.exists(self.log_laser_detector):
            with open(self.log_laser_detector, 'r') as file:
                laser_detector_dict = json.load(file)
            # Extract the laser detector time
            laser_detector_time = np.array(laser_detector_dict['cpu_time'])
        else:
            raise Exception(
                f"Laser detector logfile not found"
            )
        
        # Determine xlim
        all_times = np.concatenate([generator_time[:,1],
                                    predictor_time[:,1],
                                    camera_detector_time[:,1],
                                    laser_detector_time[:,1]])
        
        min_time = np.min(all_times)
        max_time = np.max(all_times)

        # Plot timing
        plt.figure(figsize=(10,6))

        plt.step(generator_time[:, 1], generator_time[:, 0], label='generator')
        plt.step(predictor_time[:, 1], predictor_time[:, 0], label='predictor')
        plt.step(camera_detector_time[:, 1], camera_detector_time[:, 0], label='camera')
        plt.step(laser_detector_time[:, 1], laser_detector_time[:, 0], label='laser')

        plt.title('Modules\' Iteration time')
        plt.xlabel('$t \quad [s]$')
        plt.ylabel('$iteration \ time \quad [s]$')
        plt.hlines(1 / camera_frequency, min_time, max_time, colors='red', linestyles='--')
        plt.xlim([min_time, max_time])
        plt.ylim([-0.02, 1 / camera_frequency + 0.02])
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(time_path)
        plt.show()

    def plot_camera(self):
        # Specify the saving path
        cam_path = os.path.join(self.animation_dir, self.filename + '_camera.mp4')

        # Open the camera detector log file
        if os.path.exists(self.log_camera_detector):
            with open(self.log_camera_detector, 'r') as file:
                camera_detector_dict = json.load(file)
        else:
            raise Exception(
                f"Camera detector logfile not found"
            )
        
        # Extract the camera detector data
        time = np.array(camera_detector_dict['cpu_time'])
        measurements = camera_detector_dict['measurements']
        robot_config = np.array(camera_detector_dict['robot_config'])
        b = camera_detector_dict['b']
        shooting_nodes = robot_config.shape[0]
        robot_center = np.empty((shooting_nodes, 2))
        for i in range(shooting_nodes):
            robot_center[i, 0] = robot_config[i, 0] - b * math.cos(robot_config[i, 2])
            robot_center[i, 1] = robot_config[i, 1] - b * math.sin(robot_config[i, 2])

        frequency = camera_detector_dict['frequency']
        n_points = camera_detector_dict['n_points']
        boundary_vertexes = np.array(camera_detector_dict['boundary_vertexes'])
        base_radius = camera_detector_dict['base_radius']
        simulation = camera_detector_dict['simulation']
        if simulation:
            n_agents = camera_detector_dict['n_agents']
            agents_pos = np.array(camera_detector_dict['agents_pos'])
            agent_radius = camera_detector_dict['agent_radius']

        # Plot animation with camera measurements
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(1,1)
        ax = plt.subplot(gs[0, 0])

        robot = Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='k', label='TIAGo')
        controlled_pt = ax.scatter([], [], marker='.', color='k')
        robot_label = ax.text(np.nan, np.nan, robot.get_label(), fontsize=16, ha='left', va='bottom')
        meas, = ax.plot([], [], color='magenta', marker='.', markersize=5, linestyle='', label='meas')

        boundary_line = []
        for i in range(n_points - 1):
            x_values = [boundary_vertexes[i, 0], boundary_vertexes [i + 1, 0]]
            y_values = [boundary_vertexes[i, 1], boundary_vertexes [i + 1, 1]]
            line, = ax.plot(x_values, y_values, color='red', linestyle='--')
            boundary_line.append(line)
        x_values = [boundary_vertexes[n_points - 1, 0], boundary_vertexes [0, 0]]
        y_values = [boundary_vertexes[n_points - 1, 1], boundary_vertexes [0, 1]]
        line, = ax.plot(x_values, y_values, color='red', linestyle='--')
        boundary_line.append(line)

        if simulation:
            agents = []
            agents_label = []
            agents_clearance = []
            for i in range(n_agents):
                agents.append(ax.scatter([], [], marker='.', label='hum{}'.format(i+1), color='k', alpha=0.3))
                agents_clearance.append(Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='k', linestyle='--', alpha=0.3))
                agents_label.append(ax.text(np.nan, np.nan, agents[i].get_label(), fontsize=16, ha='left', va='bottom', alpha=0.3))

        ax.set_title('TIAGo camera measurements')
        ax.set_xlabel("$x \quad [m]$")
        ax.set_ylabel('$y \quad [m]$')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

        # init and update function for the camera animation
        def init():
            robot.set_center(robot_center[0])
            robot.set_radius(base_radius)
            ax.add_patch(robot)
            controlled_pt.set_offsets(robot_config[0, :2])
            robot_label.set_position(robot_center[0])
            if simulation:
                for i in range(n_agents):
                    agent_pos = agents_pos[0, i, :]
                    agents[i].set_offsets(agent_pos)
                    agents_clearance[i].set_center(agent_pos)
                    agents_clearance[i].set_radius(agent_radius)
                    ax.add_patch(agents_clearance[i])
                    agents_label[i].set_position(agent_pos)
                return robot, robot_label, controlled_pt, agents, agents_clearance, agents_label
            return robot, robot_label, controlled_pt
        
        def update(frame):
            if frame == shooting_nodes - 1:
                animation.event_source.stop()
            ax.set_title(f'TIAGo camera measurements, t={time[frame, 1]}')
            robot.set_center(robot_center[frame])
            controlled_pt.set_offsets(robot_config[frame, :2])
            robot_label.set_position(robot_center[frame])
            current_meas = np.array(measurements[frame])
            if current_meas.shape[0] > 0:
                meas.set_data(current_meas[:, 0], current_meas[:, 1])
            else:
                meas.set_data([], [])
            if simulation:
                for i in range(n_agents):
                    agent_pos = agents_pos[frame, i, :]
                    agents[i].set_offsets(agent_pos)
                    agents_clearance[i].set_center(agent_pos)
                    agents_label[i].set_position(agent_pos)
                return robot, robot_label, controlled_pt, meas, \
                       agents, agents_clearance, agents_label
            return robot, robot_label, controlled_pt, meas

        animation = FuncAnimation(fig, update,
                                  frames=shooting_nodes,
                                  init_func=init,
                                  interval=1/frequency,
                                  blit=False,
                                  repeat=False)
        fig.tight_layout()

        if self.save_video:
            animation.save(cam_path, writer='ffmpeg', fps=frequency, dpi=80)
            print("Camera animation saved")
        
        plt.show()

    def run(self):
        self.plot_times()
        self.plot_camera()

# def plot_results(filename=None):

#         plot_camera(filename, animation_savedir)
#         plot_laser(filename, animation_savedir)
#         plot_motion(filename, log_dir, plots_savedir, animation_savedir)

#         configuration_savepath = os.path.join(plots_savedir, filename + '_configuration.png')
#         velocity_savepath = os.path.join(plots_savedir, filename + '_velocities.png')
#         acceleration_savepath = os.path.join(plots_savedir, filename + '_accelerations.png')
#         time_savepath = os.path.join(plots_savedir, filename + '_time.png')
#         kalman_savepath = os.path.join(plots_savedir, filename + '_kalman.png')
#         scans_savepath = os.path.join(animation_savedir, filename + '_scans.mp4')
#         world_savepath = os.path.join(animation_savedir, filename + '_world.mp4')
#     # Extract the generator data
#     generator_time = np.array(generator_dict['cpu_time'])
#     states = np.array(generator_dict['states'])
#     configurations = states[:, :3]
#     robot_center = np.empty((configurations.shape[0], 2))
#     b = generator_dict['offset_b']
#     for i in range(configurations.shape[0]):
#         robot_center[i, 0] = configurations[i, 0] - b * math.cos(configurations[i, 2])
#         robot_center[i, 1] = configurations[i, 1] - b * math.sin(configurations[i, 2])
    
#     driving_velocities = states[:, 3]
#     steering_velocities = states[:, 4]
#     robot_predictions = np.array(generator_dict['robot_predictions'])
#     inputs = np.array(generator_dict['wheels_accelerations'])
#     wheels_velocities = np.array(generator_dict['wheels_velocities'])
#     commanded_vel = np.array(generator_dict['commanded_velocities'])
#     targets = np.array(generator_dict['targets'])
#     errors = targets[:, :2] - configurations[:, :2]
#     wheel_radius = generator_dict['wheel_radius']
#     wheel_separation = generator_dict['wheel_separation']
#     driving_acc = wheel_radius * 0.5 * (inputs[:, 0] + inputs[:, 1])
#     steering_acc = (wheel_radius / wheel_separation) * (inputs[:, 0] - inputs[:, 1])

#     n_edges = generator_dict['n_edges']
#     boundary_vertexes = np.array(generator_dict['boundary_vertexes'])
#     input_bounds = np.array(generator_dict['input_bounds'])
#     v_bounds = np.array(generator_dict['v_bounds'])
#     omega_bounds = np.array(generator_dict['omega_bounds'])
#     wheels_vel_bounds = np.array(generator_dict['wheels_vel_bounds'])
#     vdot_bounds = np.array(generator_dict['vdot_bounds'])
#     omegadot_bounds = np.array(generator_dict['omegadot_bounds'])

#     n_actors = generator_dict['n_actors']
#     n_clusters = generator_dict['n_clusters']
#     simulation = generator_dict['simulation']
#     if n_actors > 0:
#         fake_sensing = generator_dict['fake_sensing']
#         use_kalman = generator_dict['use_kalman']
#         actors_predictions = np.array(generator_dict['actors_predictions'])
#         if simulation and not fake_sensing:
#             actors_groundtruth = np.array(generator_dict['actors_gt'])
        
#     rho_cbf = generator_dict['rho_cbf']
#     ds_cbf = generator_dict['ds_cbf']
#     frequency = generator_dict['frequency']
#     base_radius = generator_dict['base_radius']
#     shooting_nodes = inputs.shape[0]
#     t = inputs[:, 2]

#     # If the prediction module is present, open the predictor log file
#     if n_actors > 0:
#         if os.path.exists(log_predictor):
#             with open(log_predictor, 'r') as file:
#                 predictor_dict = json.load(file)
#         else:
#             raise Exception(
#                 f"Specified file not found"
#             )
#         # Extract the predictor data
#         kfs_info = predictor_dict['kfs']
#         predictor_time = np.array(predictor_dict['cpu_time'])
#         robot_states = np.array(predictor_dict['robot_states'])
#         robot_config = robot_states[:, :3]
#         agents_predictions = np.array(predictor_dict['agents_predictions'])
#         if use_kalman:
#             fsm_estimates = np.array(predictor_dict['fsm_estimates'])

#     # Open the laser detector log file
#     if os.path.exists(log_laser_detector):
#         with open(log_laser_detector, 'r') as file:
#             laser_detector_dict = json.load(file)
#     else:
#         raise Exception(
#             f"Specified file not found"
#         )

#     # Extract the laser detector data
#     laser_detector_time = np.array(laser_detector_dict['cpu_time'])
#     laser_scans = laser_detector_dict['laser_scans']
#     measurements = laser_detector_dict['measurements']
#     angle_inc = laser_detector_dict['angle_inc']
#     offset = laser_detector_dict['laser_offset']
#     angle_min = laser_detector_dict['angle_min'] + angle_inc * offset
#     angle_max = laser_detector_dict['angle_max'] - angle_inc * offset
#     range_min = laser_detector_dict['range_min']
#     range_max = laser_detector_dict['range_max']
#     laser_position = np.array(laser_detector_dict['laser_relative_pos'])       

#     # Figure representing elapsed time per module iteration
#     fig, axs = plt.subplots(3, 1, figsize=(16, 8))
    
#     # Laser detector
#     axs[0].step(laser_detector_time[:, 1], laser_detector_time[:, 0])
#     axs[0].set_title('Laser detector module iteration time')
#     axs[0].set_xlabel('$t \quad [s]$')
#     axs[0].set_ylabel('$iteration \quad time \quad [s]$')
#     axs[0].hlines(1 / frequency, laser_detector_time[0, 1], laser_detector_time[-1, 1], color='red', linestyle='--')
#     axs[0].set_xlim([laser_detector_time[0, 1], laser_detector_time[-1, 1]])
#     axs[0].grid(True) 
#     # Generator
#     axs[1].step(generator_time[:, 1], generator_time[:, 0])
#     axs[1].set_title('Generator module iteration time')
#     axs[1].set_xlabel('$t \quad [s]$')
#     axs[1].set_ylabel('$iteration \quad time \quad [s]$')
#     axs[1].hlines(1 / frequency, generator_time[0, 1], generator_time[-1, 1], color='red', linestyle='--')
#     axs[1].set_xlim([generator_time[0, 1], generator_time[-1, 1]])
#     axs[1].grid(True)
#     # Predictor
#     if n_actors > 0:
#         axs[2].step(predictor_time[:, 1], predictor_time[:, 0])
#         axs[2].set_title('Elapsed time per predictor iteration')
#         axs[2].set_xlabel('$t \quad [s]$')
#         axs[2].set_ylabel('$iteration \quad time \quad [s]$')
#         axs[2].hlines(1 / frequency, predictor_time[0, 1], predictor_time[-1, 1], color='red', linestyle='--')
#         axs[2].set_xlim([predictor_time[0, 1], predictor_time[-1, 1]])
#         axs[2].grid(True)

#     fig.tight_layout()
#     fig.savefig(time_savepath)

#     # Configuration figure
#     config_fig, config_ax = plt.subplots(4, 1, figsize=(16, 8))

#     config_ax[0].plot(t, configurations[:, 0], label='$x$')
#     config_ax[0].plot(t, targets[:, 0], label='$x_g$')
#     config_ax[1].plot(t, configurations[:, 1], label='$y$')
#     config_ax[1].plot(t, targets[:, 1], label="$y_g$")
#     config_ax[2].plot(t, errors[:, 0], label='$e_x$')
#     config_ax[2].plot(t, errors[:, 1], label='$e_y$')
#     config_ax[3].plot(t, configurations[:, 2], label='$\theta$')

#     config_ax[0].set_title('$x-position$')
#     config_ax[0].set_xlabel('$t \quad [s]$')
#     config_ax[0].set_ylabel('$[m]$')
#     config_ax[0].legend(loc='upper left')
#     config_ax[0].set_xlim([t[0], t[-1]])
#     config_ax[0].grid(True)

#     config_ax[1].set_title('$y-position$')
#     config_ax[1].set_xlabel('$t \quad [s]$')
#     config_ax[1].set_ylabel('$[m]$')
#     config_ax[1].legend(loc='upper left')
#     config_ax[1].set_xlim([t[0], t[-1]])
#     config_ax[1].grid(True)

#     config_ax[2].set_title('position errors')
#     config_ax[2].set_xlabel("$t \quad [s]$")
#     config_ax[2].set_ylabel('$[m]$')
#     config_ax[2].legend(loc='upper left')
#     config_ax[2].set_xlim([t[0], t[-1]])
#     config_ax[2].grid(True)

#     config_ax[3].set_title('TIAGo orientation')
#     config_ax[3].set_xlabel('$t \quad [s]$')
#     config_ax[3].set_ylabel('$[rad]$')
#     config_ax[3].set_ylim([-1 + np.min(configurations[:, 2]), 1 + np.max(configurations[:, 2])])
#     config_ax[3].set_xlim([t[0], t[-1]])
#     config_ax[3].grid(True)

#     config_fig.tight_layout()
#     config_fig.savefig(configuration_savepath)

#     # Velocities figure
#     vel_fig, vel_ax = plt.subplots(3, 1, figsize=(16, 8))

#     vel_ax[0].plot(t, wheels_velocities[:, 0], label='$\omega^R$')
#     vel_ax[0].plot(t, wheels_velocities[:, 1], label='$\omega^L$')
#     vel_ax[1].plot(t, driving_velocities, label='$v$')
#     vel_ax[1].plot(t, commanded_vel[:, 0], label='$v^{cmd}$')
#     vel_ax[2].plot(t, steering_velocities, label='$\omega$')
#     vel_ax[2].plot(t, commanded_vel[:, 1], label='$\omega^{cmd}$')

#     vel_ax[0].set_title('wheels velocities')
#     vel_ax[0].set_xlabel("$t \quad [s]$")
#     vel_ax[0].set_ylabel('$[rad/s]$')
#     vel_ax[0].legend(loc='upper left')
#     vel_ax[0].hlines(wheels_vel_bounds[0], t[0], t[-1], color='red', linestyle='--')
#     vel_ax[0].hlines(wheels_vel_bounds[1], t[0], t[-1], color='red', linestyle="--")
#     vel_ax[0].set_ylim([-1 + wheels_vel_bounds[0], 1 + wheels_vel_bounds[1]])
#     vel_ax[0].set_xlim([t[0], t[-1]])
#     vel_ax[0].grid(True)

#     vel_ax[1].set_title('TIAGo driving velocity')
#     vel_ax[1].set_xlabel("$t \quad [s]$")
#     vel_ax[1].set_ylabel('$[m/s]$')
#     vel_ax[1].legend(loc='upper left')
#     vel_ax[1].hlines(v_bounds[0], t[0], t[-1], color='red', linestyle='--')
#     vel_ax[1].hlines(v_bounds[1], t[0], t[-1], color='red', linestyle="--")
#     vel_ax[1].set_ylim([-1 + v_bounds[0], 1 + v_bounds[1]])
#     vel_ax[1].set_xlim([t[0], t[-1]])
#     vel_ax[1].grid(True)

#     vel_ax[2].set_title('TIAGo steering velocity')
#     vel_ax[2].set_xlabel("$t \quad [s]$")
#     vel_ax[2].set_ylabel('$[rad/s]$')
#     vel_ax[2].legend(loc='upper left')
#     vel_ax[2].hlines(omega_bounds[0], t[0], t[-1], color='red', linestyle='--')
#     vel_ax[2].hlines(omega_bounds[1], t[0], t[-1], color='red', linestyle="--")
#     vel_ax[2].set_ylim([-1 + omega_bounds[0], 1 + omega_bounds[1]])
#     vel_ax[2].set_xlim([t[0], t[-1]])
#     vel_ax[2].grid(True)

#     vel_fig.tight_layout()
#     vel_fig.savefig(velocity_savepath)

#     # Accelerations figure
#     acc_fig, acc_ax = plt.subplots(3, 1, figsize=(16, 8))

#     acc_ax[0].plot(t, inputs[:, 0], label='$\dot{\omega}^R$')
#     acc_ax[0].plot(t, inputs[:, 1], label='$\dot{\omega}^L$')
#     acc_ax[1].plot(t, driving_acc, label='$\dot{v}$')
#     acc_ax[2].plot(t, steering_acc, label='$\dot{omega}$')
    
#     acc_ax[0].set_title('wheels accelerations')
#     acc_ax[0].set_xlabel("$t \quad [s]$")
#     acc_ax[0].set_ylabel('$[rad/s^2]$')
#     acc_ax[0].legend(loc='upper left')
#     acc_ax[0].hlines(input_bounds[0], t[0], t[-1], color='red', linestyle='--')
#     acc_ax[0].hlines(input_bounds[1], t[0], t[-1], color='red', linestyle="--")
#     acc_ax[0].set_ylim([-1 + input_bounds[0], 1 + input_bounds[1]])
#     acc_ax[0].set_xlim([t[0], t[-1]])
#     acc_ax[0].grid(True)

#     acc_ax[1].set_title('TIAGo driving acceleration')
#     acc_ax[1].set_xlabel("$t \quad [s]$")
#     acc_ax[1].set_ylabel('$\dot{v} \ [m/s^2]$')
#     acc_ax[1].hlines(vdot_bounds[0], t[0], t[-1], color='red', linestyle='--')
#     acc_ax[1].hlines(vdot_bounds[1], t[0], t[-1], color='red', linestyle="--")
#     acc_ax[1].set_ylim([-1 + vdot_bounds[0], 1 + vdot_bounds[1]])
#     acc_ax[1].set_xlim([t[0], t[-1]])
#     acc_ax[1].grid(True)

#     acc_ax[2].set_title('TIAGo steering acceleration')
#     acc_ax[2].set_xlabel("$t \quad [s]$")
#     acc_ax[2].set_ylabel('$\dot{\omega} \ [rad/s^2]$')
#     acc_ax[2].hlines(omegadot_bounds[0], t[0], t[-1], color='red', linestyle='--')
#     acc_ax[2].hlines(omegadot_bounds[1], t[0], t[-1], color='red', linestyle="--")
#     acc_ax[2].set_ylim([-1 + omegadot_bounds[0], 1 + omegadot_bounds[1]])
#     acc_ax[2].set_xlim([t[0], t[-1]])
#     acc_ax[2].grid(True) 

#     acc_fig.tight_layout()
#     acc_fig.savefig(acceleration_savepath)

#     # Kalman filters figure
#     if n_actors > 0 and use_kalman:
#         n_kfs = fsm_estimates.shape[1]
#         t_predictor = predictor_time[:, 1]
#         colors = ['b', 'orange', 'g', 'r', 'c', 'purple', 'brown']
#         kfs_fig, kfs_ax = plt.subplots(3, n_kfs, figsize=(16, 8))
#         for i in range(n_kfs):
#             plot = True
#             active = False
#             plot_start_idx = 0
#             active_start_idx = 0
#             distances = np.linalg.norm(robot_config[:, :2] - fsm_estimates[:, i, :2], axis=1)
#             for j, time in enumerate(t_predictor):
#                 if kfs_info[f'KF_{i + 1}'][j][0] == 'FSMStates.ACTIVE':
#                     if not active:
#                         active = True
#                         active_start_idx = j
#                 else:
#                     if active:
#                         active = False
#                         kfs_ax[0, i].fill_between(t_predictor, 0, 14,
#                                                   where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[j]),
#                                                   color='gray', alpha=0.1)
#                         kfs_ax[1, i].fill_between(t_predictor, -1.4, 1.4,
#                                                   where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[j]),
#                                                   color='gray', alpha=0.1)
#                         kfs_ax[2, i].fill_between(t_predictor, -1.4, 1.4,
#                                                   where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[j]),
#                                                   color='gray', alpha=0.1)
                
#                 if all(pos == Hparams.nullpos for pos in fsm_estimates[j, i, :2]):
#                     if not plot:
#                         kfs_ax[0, i].plot(t_predictor[plot_start_idx : j], distances[plot_start_idx : j], color=colors[i], label='$\hat{d}$')
#                         kfs_ax[1, i].plot(t_predictor[plot_start_idx : j], fsm_estimates[plot_start_idx : j, i, 2], color=colors[i], label='$\hat{\dot{p}}_x$')
#                         kfs_ax[2, i].plot(t_predictor[plot_start_idx : j], fsm_estimates[plot_start_idx : j, i, 3], color=colors[i], label='$\hat{\dot{p}}_y$')          
#                         plot = True
#                 else:
#                     if plot:
#                         plot_start_idx = j
#                         plot = False

#             if active:
#                 kfs_ax[0, i].fill_between(t_predictor, 0, 14,
#                                           where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[-1]),
#                                           color='gray', alpha=0.1)
#                 kfs_ax[1, i].fill_between(t_predictor, -1.4, 1.4,
#                                           where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[-1]),
#                                           color='gray', alpha=0.1)
#                 kfs_ax[2, i].fill_between(t_predictor, -1.4, 1.4,
#                                           where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[-1]),
#                                           color='gray', alpha=0.1)
#             if not plot:
#                 kfs_ax[0, i].plot(t_predictor[plot_start_idx:], distances[plot_start_idx:], color=colors[i], label='$\hat{d}$')
#                 kfs_ax[1, i].plot(t_predictor[plot_start_idx:], fsm_estimates[plot_start_idx:, i, 2], color=colors[i], label='$\hat{\dot{p}}_x$')
#                 kfs_ax[2, i].plot(t_predictor[plot_start_idx:], fsm_estimates[plot_start_idx:, i, 3], color=colors[i], label='$\hat{\dot{p}}_y$')

#             kfs_ax[0, i].set_title(f'KF-{i + 1}')
#             kfs_ax[0, i].set_xlabel("$t \quad [s]$")
#             kfs_ax[0, i].set_ylabel('$\hat{d} \quad [m]$')
#             kfs_ax[0, i].hlines(rho_cbf + ds_cbf, t_predictor[0], t_predictor[-1], color='red', linestyle='--')
#             kfs_ax[1, i].set_xlabel("$t \quad [s]$")
#             kfs_ax[1, i].set_ylabel('$\hat{\dot{p}}_x \quad [m/s]$')
#             kfs_ax[2, i].set_xlabel("$t \quad [s]$")
#             kfs_ax[2, i].set_ylabel('$\hat{\dot{p}}_y \quad [m/s]$')
#             kfs_ax[0, i].set_xlim([t_predictor[0], t_predictor[-1]])
#             kfs_ax[1, i].set_xlim([t_predictor[0], t_predictor[-1]])
#             kfs_ax[2, i].set_xlim([t_predictor[0], t_predictor[-1]])
#             kfs_ax[0, i].set_ylim([0, 14])
#             kfs_ax[1, i].set_ylim([-1.4, 1.4])
#             kfs_ax[2, i].set_ylim([-1.4, 1.4])
#             kfs_ax[0, i].grid(True)
#             kfs_ax[1, i].grid(True) 
#             kfs_ax[2, i].grid(True)

#         kfs_fig.tight_layout()
#         kfs_fig.savefig(kalman_savepath)

#     plt.show()

#     # Figure to plot world animation
#     world_fig = plt.figure(figsize=(8, 8))
#     gs = gridspec.GridSpec(1,1)
#     ax_wrld = plt.subplot(gs[0, 0])

#     robot = Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='k', label='TIAGo')
#     controlled_pt = ax_wrld.scatter([], [], marker='.', color='k')
#     robot_label = ax_wrld.text(np.nan, np.nan, robot.get_label(), fontsize=16, ha='left', va='bottom')
#     robot_clearance = Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='r', linestyle='--')
#     goal = ax_wrld.scatter([], [], s=100, marker='*', label='goal', color='magenta')
#     goal_label = ax_wrld.text(np.nan, np.nan, goal.get_label(), fontsize=16, ha='left', va='bottom')
#     if n_actors > 0:
#         if not fake_sensing:
#             fov = Wedge(np.zeros(1), np.zeros(1), 0.0, 0.0, color='cyan', alpha=0.1)
#             if simulation:
#                 actors_gt = []
#                 actors_gt_label = []
#                 actors_gt_clearance = []
#                 for i in range(n_actors):
#                     actors_gt.append(ax_wrld.scatter([], [], marker='.', label='hum{}'.format(i+1), color='k', alpha=0.3))
#                     actors_gt_clearance.append(Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='k', linestyle='--', alpha=0.3))
#                     actors_gt_label.append(ax_wrld.text(np.nan, np.nan, actors_gt[i].get_label(), fontsize=16, ha='left', va='bottom', alpha=0.3))

#         actors = []
#         actors_label = []
#         actors_clearance = []
#         actors_pred_line = []
#         for i in range(n_clusters):
#             actors.append(ax_wrld.scatter([], [], marker='.', label='KF-{}'.format(i+1), color='red'))
#             actors_clearance.append(Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='red', linestyle='--'))
#             actors_label.append(ax_wrld.text(np.nan, np.nan, actors[i].get_label(), fontsize=16, ha='left', va='bottom'))
#             actor_pred_line, = ax_wrld.plot([], [], color='orange', label='actor prediction')
#             actors_pred_line.append(actor_pred_line)

#     traj_line, = ax_wrld.plot([], [], color='blue', label='trajectory')
#     robot_pred_line, = ax_wrld.plot([], [], color='green', label='prediction')
#     boundary_line = []
#     for i in range(n_edges - 1):
#         x_values = [boundary_vertexes[i, 0], boundary_vertexes [i + 1, 0]]
#         y_values = [boundary_vertexes[i, 1], boundary_vertexes [i + 1, 1]]
#         line, = ax_wrld.plot(x_values, y_values, color='red', linestyle='--')
#         boundary_line.append(line)
#     x_values = [boundary_vertexes[n_edges - 1, 0], boundary_vertexes [0, 0]]
#     y_values = [boundary_vertexes[n_edges - 1, 1], boundary_vertexes [0, 1]]
#     line, = ax_wrld.plot(x_values, y_values, color='red', linestyle='--')
#     boundary_line.append(line)

#     ax_wrld.set_title('TIAGo World')
#     ax_wrld.set_xlabel("$x \quad [m]$")
#     ax_wrld.set_ylabel('$y \quad [m]$')
#     ax_wrld.set_aspect('equal', adjustable='box')
#     ax_wrld.grid(True)

#     # init and update function for the world animation
#     def init_world():
#         robot.set_center(robot_center[0])
#         robot.set_radius(base_radius)
#         ax_wrld.add_patch(robot)
#         controlled_pt.set_offsets(configurations[0, :2])
#         robot_clearance.set_center(configurations[0, :2])
#         robot_clearance.set_radius(rho_cbf)
#         ax_wrld.add_patch(robot_clearance)
#         robot_label.set_position(robot_center[0])

#         goal.set_offsets(targets[0, :2])
#         goal_label.set_position(targets[0])
#         if n_actors > 0:        
#             for i in range(n_clusters):
#                 actor_position = actors_predictions[0, i, :, 0]
#                 actors[i].set_offsets(actor_position)
#                 actors_clearance[i].set_center(actor_position)
#                 actors_clearance[i].set_radius(ds_cbf)
#                 ax_wrld.add_patch(actors_clearance[i])
#                 actors_label[i].set_position(actor_position)

#             if not fake_sensing:
#                 ax_wrld.add_patch(fov)
#             if simulation and not fake_sensing:
#                 for i in range(n_actors):
#                     actor_gt_position = actors_groundtruth[0, i, :]
#                     actors_gt[i].set_offsets(actor_gt_position)
#                     actors_gt_clearance[i].set_center(actor_gt_position)
#                     actors_gt_clearance[i].set_radius(ds_cbf)
#                     ax_wrld.add_patch(actors_gt_clearance[i])
#                     actors_gt_label[i].set_position(actor_gt_position)

#                 return robot, robot_clearance, robot_label, goal, goal_label, \
#                        actors, actors_clearance, actors_label, fov, \
#                        actors_gt, actors_gt_clearance, actors_gt_label
#             else:
#                 return robot, robot_clearance, robot_label, goal, goal_label, \
#                        actors, actors_clearance, actors_label
#         else:
#             return robot, robot_clearance, robot_label, goal, goal_label
        
#     def update_world(frame):
#         if frame == shooting_nodes - 1:
#             world_animation.event_source.stop()

#         ax_wrld.set_title(f'TIAGo World, t={generator_time[frame, 1]}')
#         robot_prediction = robot_predictions[frame, :, :]
#         current_target = targets[frame, :]
#         traj_line.set_data(configurations[:frame + 1, 0], configurations[:frame + 1, 1])
#         robot_pred_line.set_data(robot_prediction[0, :], robot_prediction[1, :])

#         robot.set_center(robot_center[frame])
#         controlled_pt.set_offsets(configurations[frame, :2])
#         robot_clearance.set_center(configurations[frame, :2])
#         robot_label.set_position(robot_center[frame])
#         goal.set_offsets(current_target[:2])
#         goal_label.set_position(current_target)
#         if n_actors > 0:
#             for i in range(n_clusters):
#                 actor_prediction = actors_predictions[frame, i, :, :]
#                 actor_position = actor_prediction[: , 0]
#                 actors[i].set_offsets(actor_position)
#                 actors_clearance[i].set_center(actor_position)
#                 actors_label[i].set_position(actor_position)
#                 actors_pred_line[i].set_data(actor_prediction[0, :], actor_prediction[1, :])
                
#             if not fake_sensing:
#                 theta = configurations[frame, 2]
#                 current_laser_pos = configurations[frame, :2] + z_rotation(theta, laser_position)
#                 fov.set_center(current_laser_pos)
#                 fov.set_radius(range_max)
#                 fov.set_theta1((theta + angle_min) * 180 / np.pi)
#                 fov.set_theta2((theta + angle_max) * 180 / np.pi)
#                 fov.set_width(range_max - range_min)

#                 if simulation:
#                     for i in range(n_actors):
#                         actor_gt_position = actors_groundtruth[frame, i, :]
#                         actors_gt[i].set_offsets(actor_gt_position)
#                         actors_gt_clearance[i].set_center(actor_gt_position)
#                         actors_gt_label[i].set_position(actor_gt_position)

#                     return robot, robot_clearance, robot_label, goal, goal_label, \
#                         traj_line, robot_pred_line, fov, \
#                         actors, actors_clearance, actors_label, actors_pred_line, \
#                         actors_gt, actors_gt_clearance, actors_gt_label

#                 return robot, robot_clearance, robot_label, goal, goal_label, \
#                         traj_line, robot_pred_line, fov, \
#                         actors, actors_clearance, actors_label, actors_pred_line
#             else:
#                 return robot, robot_clearance, robot_label, goal, goal_label, \
#                        traj_line, robot_pred_line, \
#                        actors, actors_clearance, actors_label, actors_pred_line
#         else:
#             return robot, robot_clearance, robot_label, goal, goal_label, \
#                    traj_line, robot_pred_line
    
#     world_animation = FuncAnimation(world_fig, update_world,
#                                     frames=shooting_nodes,
#                                     init_func=init_world,
#                                     blit=False,
#                                     interval=1/frequency*500,
#                                     repeat=False)
#     world_fig.tight_layout()
#     if save_video:
#         world_animation.save(world_savepath, writer='ffmpeg', fps=frequency, dpi=80)
#         print("World animation saved")
    
#     plt.show()

#     # # Figure to plot scans animation
#     # if n_actors > 0 and not fake_sensing:
#     #     scans_fig = plt.figure(figsize=(8, 8))
#     #     gs = gridspec.GridSpec(1,1)
#     #     ax = plt.subplot(gs[0, 0])

#     #     robot = Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='k', label='TIAGo')
#     #     controlled_pt = ax.scatter([], [], marker='.', color='k')
#     #     robot_label = ax.text(np.nan, np.nan, robot.get_label(), fontsize=16, ha='left', va='bottom')
#     #     scans, = ax.plot([], [], color='magenta', marker='.', markersize=3, linestyle='', label='scans')
#     #     fov = Wedge(np.zeros(1), np.zeros(1), 0.0, 0.0, color='cyan', alpha=0.1)

#     #     core_pred_line = []
#     #     core_points_position = []
#     #     core_points_label = []        
#     #     for i in range(n_clusters):
#     #         core_points_position.append(ax.scatter([], [], marker='.', label='KF-{}'.format(i+1), color='b'))
#     #         core_points_label.append(ax.text(np.nan, np.nan, core_points_position[i].get_label(), fontsize=16, ha='left', va='bottom'))
#     #         pt_pred_line, = ax.plot([], [], color='orange', label='actor prediction')
#     #         core_pred_line.append(pt_pred_line)

#     #     boundary_line = []
#     #     for i in range(n_edges - 1):
#     #         x_values = [boundary_vertexes[i, 0], boundary_vertexes [i + 1, 0]]
#     #         y_values = [boundary_vertexes[i, 1], boundary_vertexes [i + 1, 1]]
#     #         line, = ax.plot(x_values, y_values, color='red', linestyle='--')
#     #         boundary_line.append(line)
#     #     x_values = [boundary_vertexes[n_edges - 1, 0], boundary_vertexes [0, 0]]
#     #     y_values = [boundary_vertexes[n_edges - 1, 1], boundary_vertexes [0, 1]]
#     #     line, = ax.plot(x_values, y_values, color='red', linestyle='--')
#     #     boundary_line.append(line)

#     #     ax.set_title('TIAGo Scans')
#     #     ax.set_xlabel("$x \quad [m]$")
#     #     ax.set_ylabel('$y \quad [m]$')
#     #     ax.set_aspect('equal', adjustable='box')
#     #     ax.grid(True)

#     #     shooting_nodes = robot_config.shape[0]
#     #     robot_center = np.empty((robot_config.shape[0], 2))
#     #     for i in range(robot_config.shape[0]):
#     #         robot_center[i, 0] = robot_config[i, 0] - b * math.cos(robot_config[i, 2])
#     #         robot_center[i, 1] = robot_config[i, 1] - b * math.sin(robot_config[i, 2])

#     #     # init and update function for the scans animation
#     #     def init_scans():
#     #         robot.set_center(robot_center[0])
#     #         robot.set_radius(base_radius)
#     #         ax.add_patch(robot)
#     #         ax.add_patch(fov)
#     #         controlled_pt.set_offsets(robot_config[0, :2])
#     #         robot_label.set_position(robot_center[0])

#     #         for i in range(n_clusters):
#     #             core_point_position = core_points_predictions[0, i, :, 0]
#     #             core_points_position[i].set_offsets(core_point_position)
#     #             core_points_label[i].set_position(core_point_position)
        
#     #         return robot, fov, robot_label, core_points_position, core_points_label
        
#     #     def update_scans(frame):
#     #         if frame == shooting_nodes - 1:
#     #             scans_animation.event_source.stop()

#     #         ax.set_title(f'TIAGo Scans, t={predictor_time[frame, 1]}')
#     #         robot.set_center(robot_center[frame])
#     #         controlled_pt.set_offsets(robot_config[frame, :2])
#     #         robot_label.set_position(robot_center[frame])
#     #         current_scans = np.array(laser_scans[frame])

#     #         theta = robot_config[frame, 2]
#     #         current_laser_pos = robot_config[frame, :2] + z_rotation(theta, laser_position)
#     #         fov.set_center(current_laser_pos)
#     #         fov.set_radius(range_max)
#     #         fov.set_theta1((theta + angle_min) * 180 / np.pi)
#     #         fov.set_theta2((theta + angle_max) * 180 / np.pi)
#     #         fov.set_width(range_max - range_min)

#     #         for i in range(n_clusters):
#     #             core_point_prediction = core_points_predictions[frame, i, :, :]
#     #             core_point_position = core_point_prediction[: , 0]
#     #             core_points_position[i].set_offsets(core_point_position)
#     #             core_points_label[i].set_position(core_point_position)
#     #             core_pred_line[i].set_data(core_point_prediction[0, :], core_point_prediction[1, :])

#     #         if current_scans.shape[0] > 0:
#     #             scans.set_data(current_scans[:, 0], current_scans[:, 1])
#     #         else:
#     #             scans.set_data([], [])

#     #         return robot, robot_label, fov, scans, core_points_position, core_points_label, core_pred_line

#     #     scans_animation = FuncAnimation(scans_fig, update_scans,
#     #                                     frames=shooting_nodes,
#     #                                     init_func=init_scans,
#     #                                     blit=False,
#     #                                     interval=1/frequency*500,
#     #                                     repeat=False)
#     #     scans_fig.tight_layout()
#     #     if save_video:
#     #         scans_animation.save(scans_savepath, writer='ffmpeg', fps=frequency, dpi=80)
#     #         print("Scans animation saved")
        
#     #     plt.show()

def main():
    rospy.init_node('tiago_plotter', log_level=rospy.INFO)
    rospy.loginfo('TIAGo plotter module [OK]')

    filename = rospy.get_param('/filename')
    plotter = Plotter(filename)
    plotter.run()