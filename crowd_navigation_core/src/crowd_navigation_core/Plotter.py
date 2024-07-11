import numpy as np
import rospy
import math
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Wedge
from matplotlib.patches import Polygon as Area
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
        plt.style.use('seaborn-whitegrid')

        # Specify logging directory
        log_dir = Hparams.log_dir
        if not os.path.exists(log_dir):
            raise Exception(f"Specified directory not found")

        # Set the loggers
        self.log_generator = os.path.join(log_dir, filename + '_generator.json')
        self.log_predictor = os.path.join(log_dir, filename + '_predictor.json')
        self.log_laser = os.path.join(log_dir, filename + '_laser.json')
        self.log_camera = os.path.join(log_dir, filename + '_camera.json')

        # Extract the generator dictionary
        if os.path.exists(self.log_generator):
            with open(self.log_generator, 'r') as file:
                self.generator_dict = json.load(file)
                self.perception_mode = self.generator_dict['perception']
                self.n_filters = self.generator_dict['n_filters']
        else:
            raise Exception(
                f"Generator logfile not found"
            )

        # Extract the predictor dictionary
        if self.n_filters > 0:
            if os.path.exists(self.log_predictor):
                with open(self.log_predictor, 'r') as file:
                    self.predictor_dict = json.load(file)
            else:
                raise Exception(
                    f"Predictor logfile not found"
                )

        if self.perception_mode in ('Perception.BOTH', 'Perception.CAMERA'):
            # Extract the camera detector dictionary
            if os.path.exists(self.log_camera):
                with open(self.log_camera, 'r') as file:
                    self.camera_dict = json.load(file)
            else:
                raise Exception(
                    f"Camera detector logfile not found"
                )

        if self.perception_mode in ('Perception.BOTH', 'Perception.LASER'):
            # Extract the laser detector dictionary
            if os.path.exists(self.log_laser):
                with open(self.log_laser, 'r') as file:
                    self.laser_dict = json.load(file)
            else:
                raise Exception(
                    f"Laser detector logfile not found"
                )
            
        # Specify saving plots directory
        self.plots_dir = '/tmp/crowd_navigation_tiago/plots'
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

        # Specify saving animations directory
        self.animation_dir = '/tmp/crowd_navigation_tiago/animations'
        if not os.path.exists(self.animation_dir):
            os.makedirs(self.animation_dir)

    def _plot_areas(self, ax, areas, viapoints):
        color = 'r'
        area_patches = []
        for i, vertexes in enumerate(areas):
            area = Area(vertexes,
                        closed=True,
                        fill=True,
                        facecolor=color,
                        alpha=0.05,
                        edgecolor=color,
                        linestyle='--',
                        label=f'Area {i}')
            ax.add_patch(area)
            area_patches.append(area)
        ax.scatter(viapoints[:, 0], viapoints[:, 1], marker='s', color='blue', alpha=0.1)
        return area_patches

    def _plot_walls(self, ax, walls):
        for wall_start, wall_end in walls:
            ax.plot([wall_start[0], wall_end[0]],
                    [wall_start[1], wall_end[1]],
                    'k-',
                    linewidth=2)
    
    def _plot_laser_fov(self, fov, theta, laser_pos, range_min, range_max, angle_min, angle_max):
        fov.set_center(laser_pos)
        fov.set_radius(range_max)
        fov.set_theta1((theta + angle_min) * 180 / np.pi)
        fov.set_theta2((theta + angle_max) * 180 / np.pi)
        fov.set_width(range_max - range_min)

    def _plot_camera_fov(self, fov, cam_pos, cam_angle, cam_horz_fov, min_length, max_length):
        vertexes = np.zeros((4, 2))
        min_angle = cam_angle - cam_horz_fov / 2
        max_angle = cam_angle + cam_horz_fov / 2
        vertexes[0, :] = cam_pos + np.array([math.cos(min_angle), math.sin(min_angle)]) * min_length
        vertexes[1] = cam_pos + np.array([math.cos(min_angle), math.sin(min_angle)]) * max_length
        vertexes[2] = cam_pos + np.array([math.cos(max_angle), math.sin(max_angle)]) * max_length
        vertexes[3] = cam_pos + np.array([math.cos(max_angle), math.sin(max_angle)]) * min_length
        fov.set_xy(vertexes)
    
    def _set_axis_properties(self,
                             ax,
                             xlabel,
                             ylabel,
                             title = None,
                             set_aspect = False,
                             legend = False,   
                             xlim = None,
                             ylim = None):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if set_aspect:
            ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        if legend:
            ax.legend(loc='lower left')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    def plot_times(self):
        # Specify the saving path
        time_path = os.path.join(self.plots_dir, self.filename + '_time.png')

        generator_frequency = self.generator_dict['frequency']
        generator_time = np.array(self.generator_dict['cpu_time'])
        all_times = generator_time[:, 1]
        if self.n_filters > 0:
            predictor_frequency = self.predictor_dict['frequency']
            predictor_time = np.array(self.predictor_dict['cpu_time'])
            all_times = np.concatenate([all_times, predictor_time[:, 1]])
            # Plot timing
            if self.perception_mode == 'Perception.BOTH':
                camera_frequency = self.camera_dict['frequency']
                laser_frequency = self.laser_dict['frequency']
                camera_time = np.array(self.camera_dict['cpu_time'])
                laser_time = np.array(self.laser_dict['cpu_time'])
                # Determine xlim
                all_times = np.concatenate([all_times,
                                            camera_time[:, 1],
                                            laser_time[:, 1]])

                time_fig, time_ax = plt.subplots(4, 1, figsize=(16,8))        
            elif self.perception_mode == 'Perception.LASER':
                laser_frequency = self.laser_dict['frequency']
                laser_time = np.array(self.laser_dict['cpu_time'])
                # Determine xlim
                all_times = np.concatenate([all_times,
                                            laser_time[:, 1]])
                time_fig, time_ax = plt.subplots(3, 1, figsize=(16,8))  
            elif self.perception_mode == 'Perception.CAMERA':
                camera_frequency = self.camera_dict['frequency']
                camera_time = np.array(self.camera_dict['cpu_time'])
                # Determine xlim
                all_times = np.concatenate([all_times,
                                            camera_time[:, 1]])
                time_fig, time_ax = plt.subplots(3, 1, figsize=(16,8))
            else:
                time_fig, time_ax = plt.subplots(2, 1, figsize=(16,8))
        else:
            time_fig, time_ax = plt.subplots(2, 1, figsize=(16,8))

        min_time = np.min(all_times)
        max_time = np.max(all_times)

        time_ax[0].step(generator_time[:, 1], generator_time[:, 0], label='generator')
        time_ax[0].hlines(1 / generator_frequency, min_time, max_time, colors='r', linestyles='--')
        self._set_axis_properties(time_ax[0],
                                  xlabel='$t \quad [s]$',
                                  ylabel='$iteration \ time \quad [s]$',
                                  title='Generator time',
                                  xlim=[min_time, max_time],
                                  ylim=[-0.02, 1 / generator_frequency + 0.02])

        if self.n_filters > 0:
            time_ax[1].step(predictor_time[:, 1], predictor_time[:, 0], label='predictor')
            time_ax[1].hlines(1 / predictor_frequency, min_time, max_time, colors='r', linestyles='--')
            self._set_axis_properties(time_ax[1],
                                      xlabel='$t \quad [s]$',
                                      ylabel='$iteration \ time \quad [s]$',
                                      title='Predictor time',
                                      xlim=[min_time, max_time],
                                      ylim=[-0.02, 1 / predictor_frequency + 0.02])

            if self.perception_mode == 'Perception.BOTH':
                time_ax[2].step(laser_time[:, 1], laser_time[:, 0], label='laser')
                time_ax[2].hlines(1 / laser_frequency, min_time, max_time, colors='r', linestyles='--')
                self._set_axis_properties(time_ax[2],
                                xlabel='$t \quad [s]$',
                                ylabel='$iteration \ time \quad [s]$',
                                title='Laser time',
                                xlim=[min_time, max_time],
                                ylim=[-0.02, 1 / laser_frequency + 0.02])
                time_ax[3].step(camera_time[:, 1], camera_time[:, 0], label='camera')
                time_ax[3].hlines(1 / camera_frequency, min_time, max_time, colors='r', linestyles='--')
                self._set_axis_properties(time_ax[3],
                                xlabel='$t \quad [s]$',
                                ylabel='$iteration \ time \quad [s]$',
                                title='Camera time',
                                xlim=[min_time, max_time],
                                ylim=[-0.02, 1 / camera_frequency + 0.02])
            elif self.perception_mode == 'Perception.CAMERA':
                time_ax[2].step(camera_time[:, 1], camera_time[:, 0], label='camera')
                time_ax[2].hlines(1 / camera_frequency, min_time, max_time, colors='r', linestyles='--')
                self._set_axis_properties(time_ax[2],
                                xlabel='$t \quad [s]$',
                                ylabel='$iteration \ time \quad [s]$',
                                title='Camera time',
                                xlim=[min_time, max_time],
                                ylim=[-0.02, 1 / camera_frequency + 0.02])
            elif self.perception_mode == 'Perception.LASER':
                time_ax[2].step(laser_time[:, 1], laser_time[:, 0], label='laser')
                time_ax[2].hlines(1 / laser_frequency, min_time, max_time, colors='r', linestyles='--')
                self._set_axis_properties(time_ax[2],
                                xlabel='$t \quad [s]$',
                                ylabel='$iteration \ time \quad [s]$',
                                title='Laser time',
                                xlim=[min_time, max_time],
                                ylim=[-0.02, 1 / laser_frequency + 0.02])
        time_fig.tight_layout()
        time_fig.savefig(time_path)
        plt.show()

    def plot_camera(self):
        # Specify the saving path
        cam_path = os.path.join(self.animation_dir, self.filename + '_camera.mp4')
        
        # Extract the camera detector data
        time = np.array(self.camera_dict['cpu_time'])
        measurements = self.camera_dict['measurements']
        robot_config = np.array(self.camera_dict['robot_config'])
        camera_pos = np.array(self.camera_dict['camera_position'])
        camera_angle = - np.array(self.camera_dict['camera_horz_angle']) - np.pi / 2
        b = self.camera_dict['b']
        shooting_nodes = robot_config.shape[0]
        robot_center = np.empty((shooting_nodes, 2))
        for i in range(shooting_nodes):
            robot_center[i, 0] = robot_config[i, 0] - b * math.cos(robot_config[i, 2])
            robot_center[i, 1] = robot_config[i, 1] - b * math.sin(robot_config[i, 2])

        frequency = self.camera_dict['frequency']
        base_radius = self.camera_dict['base_radius']
        simulation = self.camera_dict['simulation']
        if simulation:
            n_agents = self.camera_dict['n_agents']
            agents_pos = np.array(self.camera_dict['agents_pos'])
            agent_radius = self.camera_dict['agent_radius']
        cam_horz_fov = self.camera_dict['horz_fov']
        range_min = self.camera_dict['min_range']
        range_max = self.camera_dict['max_range']
        min_fov_length = range_min / math.cos(cam_horz_fov / 2)
        max_fov_length = range_max / math.cos(cam_horz_fov / 2)
        areas = np.array(self.generator_dict['areas'])
        walls = self.generator_dict['walls']

        # Plot animation with camera measurements
        fig, ax = plt.subplots(figsize=(8, 8))

        robot = Circle(np.zeros(2), np.zeros(1), facecolor='none', edgecolor='k', label='TIAGo')
        controlled_pt = ax.scatter([], [], marker='.', color='k')
        robot_label = ax.text(np.nan, np.nan, robot.get_label(), fontsize=16, ha='left', va='bottom')
        meas, = ax.plot([], [], color='blue', marker='.', markersize=5, linestyle='', label='meas')
        fov = Area(np.full((1, 2), np.nan), closed=True, fill=True, facecolor='purple', alpha=0.1)

        if simulation:
            agents = []
            agents_label = []
            agents_clearance = []
            for i in range(n_agents):
                agents.append(ax.scatter([], [], marker='.', label='ag{}'.format(i+1), color='k', alpha=0.3))
                agents_clearance.append(Circle(np.zeros(2), np.zeros(1), facecolor='none', edgecolor='k', linestyle='--', alpha=0.3))
                agents_label.append(ax.text(np.nan, np.nan, agents[i].get_label(), fontsize=16, ha='left', va='bottom', alpha=0.3))

        self._set_axis_properties(ax,
                                  xlabel="$x \quad [m]$",
                                  ylabel="$y \quad [m]$",
                                  title='TIAGo camera measurements',
                                  set_aspect=True)

        # init and update function for the camera animation
        def init():
            self._plot_walls(ax, walls)
            self._plot_areas(ax, areas)
            robot.set_center(robot_center[0])
            robot.set_radius(base_radius)
            ax.add_patch(robot)
            ax.add_patch(fov)
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
            self._plot_camera_fov(fov,
                                  camera_pos[frame],
                                  -camera_angle[frame],
                                  cam_horz_fov,
                                  min_fov_length,
                                  max_fov_length)
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

        if self.save_video:
            animation.save(cam_path, writer='ffmpeg', fps=frequency, dpi=80)
            print("Camera animation saved")
        
        plt.show()

    def plot_laser(self):
        # Specify the saving path
        las_path = os.path.join(self.animation_dir, self.filename + '_laser.mp4')
        
        # Extract the laser detector data
        time = np.array(self.laser_dict['cpu_time'])
        laser_scans = self.laser_dict['laser_scans']
        measurements = self.laser_dict['measurements']
        robot_config = np.array(self.laser_dict['robot_config'])
        laser_pos = np.array(self.laser_dict['laser_position'])
        b = self.laser_dict['b']
        shooting_nodes = robot_config.shape[0]
        robot_center = np.empty((shooting_nodes, 2))
        for i in range(shooting_nodes):
            robot_center[i, 0] = robot_config[i, 0] - b * math.cos(robot_config[i, 2])
            robot_center[i, 1] = robot_config[i, 1] - b * math.sin(robot_config[i, 2])

        frequency = self.laser_dict['frequency']
        base_radius = self.laser_dict['base_radius']
        simulation = self.laser_dict['simulation']
        if simulation:
            n_agents = self.laser_dict['n_agents']
            agents_pos = np.array(self.laser_dict['agents_pos'])
            agent_radius = self.laser_dict['agent_radius']
        angle_inc = self.laser_dict['angle_inc']
        laser_offset = self.laser_dict['laser_offset']
        angle_min = self.laser_dict['angle_min'] + angle_inc * laser_offset
        angle_max = self.laser_dict['angle_max'] - angle_inc * laser_offset
        range_min = self.laser_dict['range_min']
        range_max = self.laser_dict['range_max']
        areas = np.array(self.generator_dict['areas'])
        walls = self.generator_dict['walls']

        # Plot animation with laser measurements
        fig, ax = plt.subplots(figsize=(8, 8))

        robot = Circle(np.zeros(2), np.zeros(1), facecolor='none', edgecolor='k', label='TIAGo')
        controlled_pt = ax.scatter([], [], marker='.', color='k')
        robot_label = ax.text(np.nan, np.nan, robot.get_label(), fontsize=16, ha='left', va='bottom')
        scans, = ax.plot([], [], color='magenta', marker='.', markersize=3, linestyle='', label='scans') 
        meas, = ax.plot([], [], color='blue', marker='.', markersize=5, linestyle='', label='meas')
        fov = Wedge(np.zeros(1), np.zeros(1), 0.0, 0.0, color='cyan', alpha=0.1)

        if simulation:
            agents = []
            agents_label = []
            agents_clearance = []
            for i in range(n_agents):
                agents.append(ax.scatter([], [], marker='.', label='ag{}'.format(i+1), color='k', alpha=0.3))
                agents_clearance.append(Circle(np.zeros(2), np.zeros(1), facecolor='none', edgecolor='k', linestyle='--', alpha=0.3))
                agents_label.append(ax.text(np.nan, np.nan, agents[i].get_label(), fontsize=16, ha='left', va='bottom', alpha=0.3))

        self._set_axis_properties(ax,
                                  xlabel="$x \quad [m]$",
                                  ylabel="$y \quad [m]$",
                                  title='TIAGo laser measurements',
                                  set_aspect=True)

        # init and update function for the laser animation
        def init():
            self._plot_walls(ax, walls)
            self._plot_areas(ax, areas)
            robot.set_center(robot_center[0])
            robot.set_radius(base_radius)
            ax.add_patch(robot)
            ax.add_patch(fov)
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
            ax.set_title(f'TIAGo laser measurements, t={time[frame, 1]}')
            robot.set_center(robot_center[frame])
            controlled_pt.set_offsets(robot_config[frame, :2])
            robot_label.set_position(robot_center[frame])
            current_meas = np.array(measurements[frame])
            current_scans = np.array(laser_scans[frame])
            self._plot_laser_fov(fov,
                                 robot_config[frame, 2],
                                 laser_pos[frame],
                                 range_min,
                                 range_max,
                                 angle_min,
                                 angle_max)
            if current_scans.shape[0] > 0:
                scans.set_data(current_scans[:, 0], current_scans[:, 1])
            else:
                scans.set_data([], [])
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
                return robot, robot_label, controlled_pt, meas, scans, \
                       agents, agents_clearance, agents_label
            return robot, robot_label, controlled_pt, meas

        animation = FuncAnimation(fig, update,
                                  frames=shooting_nodes,
                                  init_func=init,
                                  interval=1/frequency,
                                  blit=False,
                                  repeat=False)

        if self.save_video:
            animation.save(las_path, writer='ffmpeg', fps=frequency, dpi=80)
            print("Laser animation saved")
        
        plt.show()

    def plot_estimation(self):
        kalman_savepath = os.path.join(self.plots_dir, self.filename + '_kalman.png')
    
        # Extract the predictor data
        kfs_info = self.predictor_dict['kfs']
        predictor_time = np.array(self.predictor_dict['cpu_time'])
        use_kalman = self.predictor_dict['use_kalman']
        if use_kalman:
            estimates = np.array(self.predictor_dict['estimations'])
        else:
            return

        n_kfs = estimates.shape[1]
        # Determine position limits
        all_x = np.array([])
        all_y = np.array([])
        for i in range(n_kfs):
            all_x = np.concatenate([all_x, estimates[:, i, 0]])
            all_y = np.concatenate([all_y, estimates[:, i, 1]])

        mask_x = all_x != Hparams.nullpos
        mask_y = all_y != Hparams.nullpos
        all_x = all_x[mask_x]
        all_y = all_y[mask_y]
        min_x = np.min(all_x) - 0.2
        max_x = np.max(all_x) + 0.2
        min_y = np.min(all_y) - 0.2
        max_y = np.max(all_y) + 0.2
        # Determine velocity limits
        all_x = np.array([])
        all_y = np.array([])
        for i in range(n_kfs):
            all_x = np.concatenate([all_x, estimates[:, i, 2]])
            all_y = np.concatenate([all_y, estimates[:, i, 3]])
        min_xd = np.min(all_x) - 0.2
        max_xd = np.max(all_x) + 0.2
        min_yd = np.min(all_y) - 0.2
        max_yd = np.max(all_y) + 0.2
        t_predictor = predictor_time[:, 1]
        colors = ['b', 'orange', 'g', 'r', 'c', 'purple', 'brown']
        kfs_fig, kfs_ax = plt.subplots(4, n_kfs, figsize=(16, 8))
        for i in range(n_kfs):
            plot = True
            active = False
            plot_start_idx = 0
            active_start_idx = 0
            for j, time in enumerate(t_predictor):
                if kfs_info[f'KF_{i + 1}'][j][0] == 'FSMStates.ACTIVE':
                    if not active:
                        active = True
                        active_start_idx = j
                else:
                    if active:
                        active = False
                        kfs_ax[0, i].fill_between(t_predictor, min_x, max_x,
                                                where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[j]),
                                                color='gray', alpha=0.1)
                        kfs_ax[1, i].fill_between(t_predictor, min_y, max_y,
                                                where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[j]),
                                                color='gray', alpha=0.1)
                        kfs_ax[2, i].fill_between(t_predictor, min_xd, max_xd,
                                                where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[j]),
                                                color='gray', alpha=0.1)
                        kfs_ax[3, i].fill_between(t_predictor, min_yd, max_yd,
                                                where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[j]),
                                                color='gray', alpha=0.1)
                
                if all(pos == Hparams.nullpos for pos in estimates[j, i, :2]):
                    if not plot:
                        kfs_ax[0, i].plot(t_predictor[plot_start_idx : j], estimates[plot_start_idx : j, i, 0], color=colors[i], label='$\hat{x}$')
                        kfs_ax[1, i].plot(t_predictor[plot_start_idx : j], estimates[plot_start_idx : j, i, 1], color=colors[i], label='$\hat{y}$')
                        kfs_ax[2, i].plot(t_predictor[plot_start_idx : j], estimates[plot_start_idx : j, i, 2], color=colors[i], label='$\hat{\dot{x}}$')
                        kfs_ax[3, i].plot(t_predictor[plot_start_idx : j], estimates[plot_start_idx : j, i, 3], color=colors[i], label='$\hat{\dot{y}}$')          
                        plot = True
                else:
                    if plot:
                        plot_start_idx = j
                        plot = False

            if active:
                kfs_ax[0, i].fill_between(t_predictor, min_x, max_x,
                                        where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[-1]),
                                        color='gray', alpha=0.1)
                kfs_ax[1, i].fill_between(t_predictor, min_y, max_y,
                                        where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[-1]),
                                        color='gray', alpha=0.1)
                kfs_ax[2, i].fill_between(t_predictor, min_xd, max_xd,
                                        where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[-1]),
                                        color='gray', alpha=0.1)
                kfs_ax[3, i].fill_between(t_predictor, min_yd, max_yd,
                                        where=(t_predictor >= t_predictor[active_start_idx]) & (t_predictor < t_predictor[-1]),
                                        color='gray', alpha=0.1)
            if not plot:
                kfs_ax[0, i].plot(t_predictor[plot_start_idx:], estimates[plot_start_idx:, i, 0], color=colors[i], label='$\hat{x}$')
                kfs_ax[1, i].plot(t_predictor[plot_start_idx:], estimates[plot_start_idx:, i, 1], color=colors[i], label='$\hat{y}$')
                kfs_ax[2, i].plot(t_predictor[plot_start_idx:], estimates[plot_start_idx:, i, 2], color=colors[i], label='$\hat{\dot{x}}$')
                kfs_ax[3, i].plot(t_predictor[plot_start_idx:], estimates[plot_start_idx:, i, 3], color=colors[i], label='$\hat{\dot{y}}$')

            self._set_axis_properties(kfs_ax[0, i],
                                      xlabel="$t \quad [s]$",
                                      ylabel='$\hat{x} \quad [m]$',
                                      title=f'KF-{i + 1}',
                                      xlim=[t_predictor[0], t_predictor[-1]],
                                      ylim=[min_x, max_x])
            self._set_axis_properties(kfs_ax[1, i],
                                      xlabel="$t \quad [s]$",
                                      ylabel='$\hat{y} \quad [m]$',
                                      xlim=[t_predictor[0], t_predictor[-1]],
                                      ylim=[min_y, max_y])
            self._set_axis_properties(kfs_ax[2, i],
                                      xlabel="$t \quad [s]$",
                                      ylabel='$\hat{\dot{x}} \quad [m/s]$',
                                      xlim=[t_predictor[0], t_predictor[-1]],
                                      ylim=[min_xd, max_xd])
            self._set_axis_properties(kfs_ax[3, i],
                                      xlabel="$t \quad [s]$",
                                      ylabel='$\hat{\dot{y}} \quad [m/s]$',
                                      xlim=[t_predictor[0], t_predictor[-1]],
                                      ylim=[min_yd, max_yd])

        kfs_fig.tight_layout()
        kfs_fig.savefig(kalman_savepath)

        plt.show()

    def plot_motion(self):
        configuration_savepath = os.path.join(self.plots_dir, self.filename + '_configuration.png')
        velocity_savepath = os.path.join(self.plots_dir, self.filename + '_velocities.png')
        acceleration_savepath = os.path.join(self.plots_dir, self.filename + '_accelerations.png')
        motion_savepath = os.path.join(self.animation_dir, self.filename + '_motion.mp4')

        # Extract the generator data
        generator_time = np.array(self.generator_dict['cpu_time'])
        robot_states = np.array(self.generator_dict['robot_state'])
        configurations = robot_states[:, :3]
        robot_center = np.empty((configurations.shape[0], 2))
        b = self.generator_dict['b']
        for i in range(configurations.shape[0]):
            robot_center[i, 0] = configurations[i, 0] - b * math.cos(configurations[i, 2])
            robot_center[i, 1] = configurations[i, 1] - b * math.sin(configurations[i, 2])
        robot_predictions = np.array(self.generator_dict['robot_predictions'])

        wheel_radius = self.generator_dict['wheel_radius']
        wheel_separation = self.generator_dict['wheel_separation']
        wheels_velocities = np.array(self.generator_dict['wheels_velocities'])
        v_actual = wheel_radius * 0.5 * (wheels_velocities[:, 0] + wheels_velocities[:, 1])
        omega_actual = (wheel_radius / wheel_separation) * (wheels_velocities[:, 0] - wheels_velocities[:, 1])
        commands = np.array(self.generator_dict['commands'])
        targets = np.array(self.generator_dict['targets'])
        target_viapoints = np.array(self.generator_dict['target_viapoints'])
        area_index = np.array(self.generator_dict['area_index'])
        errors = targets[:, :2] - configurations[:, :2]
        inputs = np.array(self.generator_dict['inputs'])
        driving_acc = wheel_radius * 0.5 * (inputs[:, 0] + inputs[:, 1])
        steering_acc = (wheel_radius / wheel_separation) * (inputs[:, 0] - inputs[:, 1])
        areas = self.generator_dict['areas']
        viapoints = np.array(self.generator_dict['viapoints'])
        walls = self.generator_dict['walls']
        input_bounds = np.array(self.generator_dict['input_bounds'])
        v_bounds = np.array(self.generator_dict['v_bounds'])
        omega_bounds = np.array(self.generator_dict['omega_bounds'])
        wheels_vel_bounds = np.array(self.generator_dict['wheels_vel_bounds'])
        vdot_bounds = np.array(self.generator_dict['vdot_bounds'])
        omegadot_bounds = np.array(self.generator_dict['omegadot_bounds'])

        if self.n_filters > 0:
            agents_predictions = np.array(self.generator_dict['agents_predictions'])
        simulation = self.generator_dict['simulation']
        if simulation:
            n_agents = self.generator_dict['n_agents']
            agents_pos = np.array(self.generator_dict['agents_pos'])
        robot_radius = self.generator_dict['robot_radius']
        agent_radius = self.generator_dict['agent_radius']
        frequency = self.generator_dict['frequency']
        base_radius = self.generator_dict['base_radius']
        dt = self.generator_dict['dt']
        N_horizon = self.generator_dict['N_horizon']
        laser_rel_pos = np.array(self.generator_dict['laser_rel_pos'])
        shooting_nodes = inputs.shape[0]
        t = inputs[:, 2]

        if self.perception_mode in ('Perception.LASER', 'Perception.BOTH'):
            laser_range_max = self.laser_dict['range_max']
            laser_range_min = self.laser_dict['range_min']
            angle_inc = self.laser_dict['angle_inc']
            laser_offset = self.laser_dict['laser_offset']
            angle_min = self.laser_dict['angle_min'] + angle_inc * laser_offset
            angle_max = self.laser_dict['angle_max'] - angle_inc * laser_offset

        if self.perception_mode in ('Perception.CAMERA', 'Perception.BOTH'):
            camera_range_max = self.camera_dict['max_range']
            camera_range_min = self.camera_dict['min_range']
            cam_horz_fov = self.camera_dict['horz_fov']
            min_fov_length = 2 * camera_range_min / cam_horz_fov
            max_fov_length = 2 * camera_range_max / cam_horz_fov 
            camera_pos = np.array(self.generator_dict['camera_position'])
            camera_angle = - np.array(self.generator_dict['camera_horz_angle']) - np.pi / 2

        # Configuration figure
        config_fig, config_ax = plt.subplots(4, 1, figsize=(16, 8))

        config_ax[0].plot(t, configurations[:, 0], label='$x$')
        config_ax[0].plot(t, targets[:, 0], label='$x_g$')
        config_ax[1].plot(t, configurations[:, 1], label='$y$')
        config_ax[1].plot(t, targets[:, 1], label="$y_g$")
        config_ax[2].plot(t, errors[:, 0], label='$e_x$')
        config_ax[2].plot(t, errors[:, 1], label='$e_y$')
        config_ax[3].plot(t, configurations[:, 2], label='$\theta$')

        
        self._set_axis_properties(config_ax[0],
                                  title='x-position',
                                  xlabel='$t \quad [s]$',
                                  ylabel='$[m]$',
                                  legend=True,
                                  xlim=[t[0], t[-1]])
        self._set_axis_properties(config_ax[1],
                                  title='y-position',
                                  xlabel='$t \quad [s]$',
                                  ylabel='$[m]$',
                                  legend=True,
                                  xlim=[t[0], t[-1]])
        self._set_axis_properties(config_ax[2],
                                  title='position errors',
                                  xlabel='$t \quad [s]$',
                                  ylabel='$[m]$',
                                  legend=True,
                                  xlim=[t[0], t[-1]])
        self._set_axis_properties(config_ax[3],
                                  title='TIAGo orientation',
                                  xlabel='$t \quad [s]$',
                                  ylabel='$[rad]$',
                                  xlim=[t[0], t[-1]],
                                  ylim=[-1 + np.min(configurations[:, 2]), 1 + np.max(configurations[:, 2])])
        config_fig.tight_layout()
        config_fig.savefig(configuration_savepath)

        # Velocities figure
        vel_fig, vel_ax = plt.subplots(3, 1, figsize=(16, 8))

        vel_ax[0].plot(t, wheels_velocities[:, 0], label='$\omega^R$')
        vel_ax[0].plot(t, wheels_velocities[:, 1], label='$\omega^L$')
        vel_ax[0].hlines(wheels_vel_bounds[0], t[0], t[-1], color='red', linestyle='--')
        vel_ax[0].hlines(wheels_vel_bounds[1], t[0], t[-1], color='red', linestyle="--")
        self._set_axis_properties(vel_ax[0],
                                  title='wheels velocities',
                                  xlabel="$t \quad [s]$",
                                  ylabel='$[rad/s]$',
                                  legend=True,
                                  xlim=[t[0], t[-1]],
                                  ylim=[-1 + wheels_vel_bounds[0], 1 + wheels_vel_bounds[1]])
        vel_ax[1].plot(t, v_actual, label='$v$')
        vel_ax[1].plot(t, commands[:, 0], label='$v^{cmd}$')
        vel_ax[1].hlines(v_bounds[0], t[0], t[-1], color='red', linestyle='--')
        vel_ax[1].hlines(v_bounds[1], t[0], t[-1], color='red', linestyle="--")
        self._set_axis_properties(vel_ax[1],
                                  title='TIAGo driving velocity',
                                  xlabel="$t \quad [s]$",
                                  ylabel='$[m/s]$',
                                  legend=True,
                                  xlim=[t[0], t[-1]],
                                  ylim=[-1 + v_bounds[0], 1 + v_bounds[1]])
        vel_ax[2].plot(t, omega_actual, label='$\omega$')
        vel_ax[2].plot(t, commands[:, 1], label='$\omega^{cmd}$')
        vel_ax[2].hlines(omega_bounds[0], t[0], t[-1], color='red', linestyle='--')
        vel_ax[2].hlines(omega_bounds[1], t[0], t[-1], color='red', linestyle="--")
        self._set_axis_properties(vel_ax[2],
                                  title='TIAGo steering velocity',
                                  xlabel="$t \quad [s]$",
                                  ylabel='$[rad/s]$',
                                  legend=True,
                                  xlim=[t[0], t[-1]],
                                  ylim=[-1 + omega_bounds[0], 1 + omega_bounds[1]])
        vel_fig.tight_layout()
        vel_fig.savefig(velocity_savepath)

        # Accelerations figure
        acc_fig, acc_ax = plt.subplots(3, 1, figsize=(16, 8))

        acc_ax[0].plot(t, inputs[:, 0], label='$\dot{\omega}^R$')
        acc_ax[0].plot(t, inputs[:, 1], label='$\dot{\omega}^L$')
        acc_ax[0].hlines(input_bounds[0], t[0], t[-1], color='red', linestyle='--')
        acc_ax[0].hlines(input_bounds[1], t[0], t[-1], color='red', linestyle="--")
        self._set_axis_properties(acc_ax[0],
                                  title='wheels accelerations',
                                  xlabel="$t \quad [s]$",
                                  ylabel='$[rad/s^2]$',
                                  legend=True,
                                  xlim=[t[0], t[-1]],
                                  ylim=[-1 + input_bounds[0], 1 + input_bounds[1]])
        acc_ax[1].plot(t, driving_acc, label='$\dot{v}$')
        acc_ax[1].hlines(vdot_bounds[0], t[0], t[-1], color='red', linestyle='--')
        acc_ax[1].hlines(vdot_bounds[1], t[0], t[-1], color='red', linestyle="--")
        self._set_axis_properties(acc_ax[1],
                                  title='TIAGo driving acceleration',
                                  xlabel="$t \quad [s]$",
                                  ylabel='$\dot{v} \ [m/s^2]$',
                                  xlim=[t[0], t[-1]],
                                  ylim=[-1 + vdot_bounds[0], 1 + vdot_bounds[1]])
        acc_ax[2].plot(t, steering_acc, label='$\dot{omega}$')
        acc_ax[2].hlines(omegadot_bounds[0], t[0], t[-1], color='red', linestyle='--')
        acc_ax[2].hlines(omegadot_bounds[1], t[0], t[-1], color='red', linestyle="--")
        self._set_axis_properties(acc_ax[2],
                                  title='TIAGo steering acceleration',
                                  xlabel="$t \quad [s]$",
                                  ylabel='$\dot{omega} \ [rad/s^2]$',
                                  xlim=[t[0], t[-1]],
                                  ylim=[-1 + omegadot_bounds[0], 1 + omegadot_bounds[1]])
        acc_fig.tight_layout()
        acc_fig.savefig(acceleration_savepath)

        plt.show()

        # Figure to plot motion animation
        fig, ax = plt.subplots(figsize=(8, 8))

        robot = Circle(np.zeros(2), np.zeros(1), facecolor='none', edgecolor='k', label='TIAGo')
        controlled_pt = ax.scatter([], [], marker='.', color='k')
        robot_label = ax.text(np.nan, np.nan, robot.get_label(), fontsize=16, ha='left', va='bottom')
        robot_clearance = Circle(np.zeros(2), np.zeros(1), facecolor='none', edgecolor='r', linestyle='--')
        goal = ax.scatter([], [], s=100, marker='*', label='goal', color='magenta')
        target_viapoint = ax.scatter([], [], marker='s', color='red', alpha=0.5)
        goal_label = ax.text(np.nan, np.nan, goal.get_label(), fontsize=16, ha='left', va='bottom')
        if self.n_filters > 0:
            if simulation:
                agents = []
                agents_label = []
                agents_clearance = []
                for i in range(n_agents):
                    agents.append(ax.scatter([], [], marker='.', label='ag{}'.format(i+1), color='k', alpha=0.3))
                    agents_clearance.append(Circle(np.zeros(2), np.zeros(1), facecolor='none', edgecolor='k', linestyle='--', alpha=0.3))
                    agents_label.append(ax.text(np.nan, np.nan, agents[i].get_label(), fontsize=16, ha='left', va='bottom', alpha=0.3))
            if self.perception_mode in ('Perception.LASER', 'Perception.BOTH'):
                laser_fov = Wedge(np.zeros(1), np.zeros(1), 0.0, 0.0, color='cyan', alpha=0.1)
            if self.perception_mode in ('Perception.CAMERA', 'Perception.BOTH'):
                camera_fov = Area(np.full((1, 2), np.nan), closed=True, fill=True, facecolor='purple', alpha=0.1)
            estimates = []
            estimates_label = []
            estimates_clearance = []
            predictions = []
            for i in range(self.n_filters):
                estimates.append(ax.scatter([], [], marker='.', label='KF-{}'.format(i+1), color='red'))
                estimates_clearance.append(Circle(np.zeros(2), np.zeros(1), facecolor='none', edgecolor='red', linestyle='--'))
                estimates_label.append(ax.text(np.nan, np.nan, estimates[i].get_label(), fontsize=16, ha='left', va='bottom'))
                prediction, = ax.plot([], [], color='orange', label='actor prediction', linewidth=agent_radius*60, alpha=0.4)
                predictions.append(prediction)
                
        traj_line, = ax.plot([], [], color='blue', label='trajectory')
        robot_pred_line, = ax.plot([], [], color='green', label='prediction', linewidth=robot_radius*60, alpha=0.4)

        self._set_axis_properties(ax,
                                  xlabel="$x \quad [m]$",
                                  ylabel="$y \quad [m]$",
                                  title='TIAGo motion',
                                  set_aspect=True)

        self._plot_walls(ax, walls)
        area_patches = self._plot_areas(ax, areas, viapoints)
        # init and update function for the motion animation
        def init_motion():
            robot.set_center(robot_center[0])
            robot.set_radius(base_radius)
            ax.add_patch(robot)
            controlled_pt.set_offsets(configurations[0, :2])
            robot_clearance.set_center(configurations[0, :2])
            robot_clearance.set_radius(robot_radius)
            ax.add_patch(robot_clearance)
            robot_label.set_position(robot_center[0])

            goal.set_offsets(targets[0, :2])
            goal_label.set_position(targets[0])
            if targets[0, 0] != target_viapoints[0, 0] and targets[0, 1] != target_viapoints[0, 1]:
                target_viapoint.set_offsets(target_viapoints[0])
            area_patches[area_index[0]].set_alpha(0.2)
            if self.n_filters > 0:        
                for i in range(self.n_filters):
                    ax.add_patch(estimates_clearance[i])
                if self.perception_mode in ('Perception.LASER', 'Perception.BOTH'):
                    ax.add_patch(laser_fov)
                if self.perception_mode in ('Perception.CAMERA', 'Perception.BOTH'):
                    ax.add_patch(camera_fov)
                if simulation:
                    for i in range(n_agents):
                        agent_pos = agents_pos[0, i, :]
                        agents[i].set_offsets(agent_pos)
                        agents_clearance[i].set_center(agent_pos)
                        agents_clearance[i].set_radius(agent_radius)
                        ax.add_patch(agents_clearance[i])
                        agents_label[i].set_position(agent_pos)
                    return robot, robot_clearance, robot_label, area_patches, \
                            controlled_pt, goal, goal_label, target_viapoint, \
                            estimates, estimates_clearance, estimates_label, \
                            agents, agents_clearance, agents_label
                return robot, robot_clearance, robot_label, area_patches, \
                       controlled_pt, goal, goal_label, target_viapoint, \
                       estimates, estimates_clearance, estimates_label
            return robot, robot_clearance, robot_label, area_patches, controlled_pt, goal, goal_label, target_viapoint
            
        def update_motion(frame):
            if frame == shooting_nodes - 1:
                motion_animation.event_source.stop()

            ax.set_title(f'TIAGo motion, t={generator_time[frame, 1]}')
            robot_prediction = robot_predictions[frame, :, :]
            current_target = targets[frame, :]
            current_viapoint = target_viapoints[frame]
            traj_line.set_data(configurations[:frame + 1, 0], configurations[:frame + 1, 1])
            robot_pred_line.set_data(robot_prediction[:, 0], robot_prediction[:, 1])

            robot.set_center(robot_center[frame])
            controlled_pt.set_offsets(configurations[frame, :2])
            robot_clearance.set_center(configurations[frame, :2])
            robot_label.set_position(robot_center[frame])
            goal.set_offsets(current_target[:2])
            goal_label.set_position(current_target)
            if current_viapoint[0] != current_target[0] and current_viapoint[1] != current_target[1]:
                target_viapoint.set_visible(True)
                target_viapoint.set_offsets(target_viapoints[frame])
            else:
                target_viapoint.set_visible(False)
            area_patches[area_index[frame]].set_alpha(0.2)
            for patch in area_patches[:area_index[frame]]:
                patch.set_alpha(0.05)
            for patch in area_patches[area_index[frame] + 1:]:
                patch.set_alpha(0.05)
            if self.n_filters > 0:
                for i in range(self.n_filters):
                    agent_estimate = agents_predictions[frame, i, :2]
                    if all(coord != Hparams.nullpos for coord in agent_estimate):
                        agent_vel_estimate = agents_predictions[frame, i, 2:]
                        estimates[i].set_offsets(agent_estimate)
                        estimates[i].set_visible(True)
                        estimates_clearance[i].set_center(agent_estimate)
                        estimates_clearance[i].set_radius(agent_radius)
                        estimates_label[i].set_position(agent_estimate)
                        estimates_label[i].set_visible(True)
                        agent_prediction = np.vstack((agent_estimate, agent_estimate + agent_vel_estimate * dt * N_horizon))
                        predictions[i].set_data(agent_prediction[:, 0], agent_prediction[:, 1])
                        estimates_label[i].set_visible(True)
                    else:
                        estimates[i].set_visible(False)
                        estimates_clearance[i].set_radius(0.0)
                        estimates_label[i].set_visible(False)
                        predictions[i].set_data([],[])
                if self.perception_mode in ('Perception.LASER', 'Perception.BOTH'):
                    current_theta = configurations[frame, 2]
                    current_laser_pos = configurations[frame, :2] + z_rotation(current_theta, laser_rel_pos)
                    self._plot_laser_fov(laser_fov,
                                         current_theta,
                                         current_laser_pos,
                                         laser_range_min,
                                         laser_range_max,
                                         angle_min,
                                         angle_max)
                if self.perception_mode in ('Perception.CAMERA', 'Perception.BOTH'):
                    self._plot_camera_fov(camera_fov,
                                          camera_pos[frame],
                                          -camera_angle[frame],
                                          cam_horz_fov,
                                          min_fov_length,
                                          max_fov_length)
                if simulation:
                    for i in range(n_agents):
                        agent_pos = agents_pos[frame, i, :]
                        agents[i].set_offsets(agent_pos)
                        agents_clearance[i].set_center(agent_pos)
                        agents_label[i].set_position(agent_pos)       
                    return robot, robot_clearance, robot_label, goal, goal_label, target_viapoint, area_patches, \
                            traj_line, robot_pred_line, \
                            agents, agents_clearance, agents_label, \
                            estimates, estimates_clearance, estimates_label, prediction
                return robot, robot_clearance, robot_label, goal, goal_label, target_viapoint, area_patches, \
                        traj_line, robot_pred_line, \
                        estimates, estimates_clearance, estimates_label, prediction
            return robot, robot_clearance, robot_label, goal, goal_label, target_viapoint, area_patches, \
                traj_line, robot_pred_line
        
        motion_animation = FuncAnimation(fig, update_motion,
                                        frames=shooting_nodes,
                                        init_func=init_motion,
                                        blit=False,
                                        interval=1/frequency*500,
                                        repeat=False)

        if self.save_video:
            motion_animation.save(motion_savepath, writer='ffmpeg', fps=frequency, dpi=80)
            print("Motion animation saved")
        
        plt.show()

    def run(self):
        self.plot_times()
        if self.n_filters > 0:
            if self.perception_mode == 'Perception.BOTH':
                self.plot_camera()
                self.plot_laser()
            elif self.perception_mode == 'Perception.LASER':
                self.plot_laser()
            elif self.perception_mode == 'Perception.CAMERA':
                self.plot_camera()
        self.plot_estimation()
        self.plot_motion()

def main():
    rospy.init_node('tiago_plotter', anonymous=True, log_level=rospy.INFO)
    rospy.loginfo('TIAGo plotter module [OK]')

    filename = rospy.get_param('/filename')
    plotter = Plotter(filename)
    plotter.run()