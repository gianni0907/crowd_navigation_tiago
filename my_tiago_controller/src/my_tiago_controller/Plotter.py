import numpy as np
import rospy
import math
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from  matplotlib.animation import FuncAnimation

def plot_results(filename=None):
    # Specify logging directory
    log_dir = '/tmp/crowd_navigation_tiago/data'
    if not os.path.exists(log_dir):
        raise Exception(
           f"Specified directory not found"
        )

    # Specify saving plots directory
    save_plots_dir = '/tmp/crowd_navigation_tiago/plots'
    if not os.path.exists(save_plots_dir):
        os.makedirs(save_plots_dir)

    # Specify saving animations directory
    save_sim_dir = '/tmp/crowd_navigation_tiago/simulations'
    if not os.path.exists(save_sim_dir):
        os.makedirs(save_sim_dir)

    log_path = os.path.join(log_dir, filename + '.json')
    save_profiles_path = os.path.join(save_plots_dir, filename + '_profiles.png')
    save_sim_path = os.path.join(save_sim_dir, filename + '_simulation.mp4')

    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            data = json.load(file)
    else:
        raise Exception(
            f"Specified file not found"
        )

    # Setup all the data
    configurations = np.array(data['configurations'])
    robot_center = np.empty((configurations.shape[0], 2))
    inputs = np.array(data['inputs'])
    predictions = np.array(data['predictions'])
    velocities = np.array(data['velocities'])
    targets = np.array(data['targets'])
    control_bounds = np.array(data['control_bounds'])
    x_bounds = np.array(data['x_bounds'])
    y_bounds = np.array(data['y_bounds'])
    v_bounds = np.array(data['v_bounds'])
    omega_bounds = np.array(data['omega_bounds'])
    n_obstacles = data['n_obstacles']
    if n_obstacles > 0:
        obstacles_position = np.array(data['humans_position'])
    rho_cbf = data['rho_cbf']
    ds_cbf = data['ds_cbf']
    frequency = data['frequency']
    b = data['offset_b']
    for i in range(configurations.shape[0]):
        robot_center[i, 0] = configurations[i, 0] - b * math.cos(configurations[i, 2])
        robot_center[i, 1] = configurations[i, 1] - b * math.sin(configurations[i, 2])
    shooting_nodes = inputs.shape[0]
    t = inputs[:, 2]

    # Figure to plot velocities
    vel_fig, axs = plt.subplots(3, 2, figsize=(16, 8))

    axs[0, 0].plot(t, inputs[:, 0], label='$\omega_r$')
    axs[0, 0].plot(t, inputs[:, 1], label='$\omega_l$')
    axs[1, 0].plot(t, velocities[:, 0], label='$v$')
    axs[2, 0].plot(t, velocities[:, 1], label='$\omega$')
    axs[0, 1].plot(t, targets[:, 0] - configurations[:, 0], label='$x$')
    axs[1, 1].plot(t, targets[:, 1] - configurations[:, 1], label='$y$')
    axs[2, 1].plot(t, configurations[:, 2], label='$\theta$')

    axs[0, 0].set_title('wheels angular velocity')
    axs[0, 0].set_xlabel("$t \quad [s]$")
    axs[0, 0].set_ylabel('$[rad/s]$')
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].hlines(control_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs[0, 0].hlines(control_bounds[1], t[0], t[-1], color='red', linestyle="--")
    axs[0, 0].set_ylim([-1 + control_bounds[0], 1 + control_bounds[1]])
    axs[0, 0].set_xlim([t[0], t[-1]])
    axs[0, 0].grid(True)

    axs[1, 0].set_title('TIAGo driving velocity')
    axs[1, 0].set_xlabel("$t \quad [s]$")
    axs[1, 0].set_ylabel('$[m/s]$')
    axs[1, 0].hlines(v_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs[1, 0].hlines(v_bounds[1], t[0], t[-1], color='red', linestyle="--")
    axs[1, 0].set_ylim([-1 + v_bounds[0], 1 + v_bounds[1]])
    axs[1, 0].set_xlim([t[0], t[-1]])
    axs[1, 0].grid(True)

    axs[2, 0].set_title('TIAGo steering velocity')
    axs[2, 0].set_xlabel("$t \quad [s]$")
    axs[2, 0].set_ylabel('$[rad/s]$')
    axs[2, 0].hlines(omega_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs[2, 0].hlines(omega_bounds[1], t[0], t[-1], color='red', linestyle="--")
    axs[2, 0].set_ylim([-1 + omega_bounds[0], 1 + omega_bounds[1]])
    axs[2, 0].set_xlim([t[0], t[-1]])
    axs[2, 0].grid(True)

    axs[0, 1].set_title('$e_x = x_g - x$')
    axs[0, 1].set_xlabel('$t \quad [s]$')
    axs[0, 1].set_ylabel('$[m]$')
    axs[0, 1].hlines(x_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs[0, 1].hlines(x_bounds[1], t[0], t[-1], color='red', linestyle='--')
    axs[0, 1].set_ylim([-1 + x_bounds[0], 1 + x_bounds[1]])
    axs[0, 1].set_xlim([t[0], t[-1]])
    axs[0, 1].grid(True)

    axs[1, 1].set_title('$e_y = y_g - y$')
    axs[1, 1].set_xlabel('$t \quad [s]$')
    axs[1, 1].set_ylabel('$[m]$')
    axs[1, 1].hlines(y_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs[1, 1].hlines(y_bounds[1], t[0], t[-1], color='red', linestyle='--')
    axs[1, 1].set_ylim([-1 + y_bounds[0], 1 + y_bounds[1]])
    axs[1, 1].set_xlim([t[0], t[-1]])
    axs[1, 1].grid(True)

    axs[2, 1].set_title('TIAGo orientation')
    axs[2, 1].set_xlabel('$t \quad [s]$')
    axs[2, 1].set_ylabel('$[rad]$')
    axs[2, 1].set_ylim([-1 + np.min(configurations[:, 2]), 1 + np.max(configurations[:, 2])])
    axs[2, 1].set_xlim([t[0], t[-1]])
    axs[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_profiles_path)
    plt.show()

    # Figure to plot simulation
    sim_fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(3,2)
    ax_big = plt.subplot(gs[:, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 1])
    ax3 = plt.subplot(gs[2, 1])
    robot = ax_big.scatter([], [], s=100.0, marker='o',label='TIAGo', facecolors='none', edgecolors='blue')
    controlled_pt = ax_big.scatter([], [], marker='.', color='blue')
    robot_label = ax_big.text(np.nan, np.nan, robot.get_label(), fontsize=8, ha='left', va='bottom')
    robot_clearance = Circle(np.nan, np.nan, facecolor='none', edgecolor='blue')
    goal = ax_big.scatter([], [], s=80.0, marker='*', label='goal', color='magenta', alpha=0.7)
    goal_label = ax_big.text(np.nan, np.nan, goal.get_label(), fontsize=8, ha='left', va='bottom')
    if n_obstacles > 0:
        obstacles = []
        obstacles_label = []
        obstacles_clearance = []
    for i in range(n_obstacles):
        obstacles.append(ax_big.scatter([], [], marker='o', label='human{}'.format(i+1), color='red', alpha=0.7))
        obstacles_clearance.append(Circle(np.nan, np.nan, facecolor='none', edgecolor='red'))
        obstacles_label.append(ax_big.text(np.nan, np.nan, obstacles[i].get_label(), fontsize=8, ha='left', va='bottom'))
    traj_line, = ax_big.plot([], [], color='blue', label='trajectory')
    pred_line, = ax_big.plot([], [], color='green', label='prediction')
    wr_line, = ax1.plot([], [], label='$\omega_r$')
    wl_line, = ax1.plot([], [], label='$\omega_l$')
    v_line, = ax2.plot([], [], label='$v$')
    omega_line, = ax3.plot([], [], label='$\omega$')

    ax_big.set_title('Simulation')
    ax_big.set_xlabel("$x \quad [m]$")
    ax_big.set_ylabel('$y \quad [m]$')
    ax_big.axhline(y_bounds[0], color='red', linestyle='--')
    ax_big.axhline(y_bounds[1], color='red', linestyle="--")
    ax_big.axvline(x_bounds[0], color='red', linestyle='--')
    ax_big.axvline(x_bounds[1], color='red', linestyle='--')
    ax_big.set_ylim([-1 + y_bounds[0], 1 + y_bounds[1]])
    ax_big.set_xlim([-1 + x_bounds[0], 1 + x_bounds[1]])
    ax_big.set_aspect('equal', adjustable='box')
    ax_big.grid(True)

    ax1.set_title('wheels angular velocity')
    ax1.set_xlabel("$t \quad [s]$")
    ax1.set_ylabel('$[rad/s]$')
    ax1.legend(loc='upper right')
    ax1.hlines(control_bounds[0], t[0], t[-1], color='red', linestyle='--')
    ax1.hlines(control_bounds[1], t[0], t[-1], color='red', linestyle="--")
    ax1.set_ylim([-1 + control_bounds[0], 1 + control_bounds[1]])
    ax1.set_xlim([t[0], t[-1]])
    ax1.grid(True)

    ax2.set_title('TIAGo driving velocity')
    ax2.set_xlabel("$t \quad [s]$")
    ax2.set_ylabel('$[m/s]$')
    ax2.hlines(v_bounds[0], t[0], t[-1], color='red', linestyle='--')
    ax2.hlines(v_bounds[1], t[0], t[-1], color='red', linestyle="--")
    ax2.set_ylim([-1 + v_bounds[0], 1 + v_bounds[1]])
    ax2.set_xlim([t[0], t[-1]])
    ax2.grid(True)

    ax3.set_title('TIAGo steering velocity')
    ax3.set_xlabel("$t \quad [s]$")
    ax3.set_ylabel('$[rad/s]$')
    ax3.hlines(omega_bounds[0], t[0], t[-1], color='red', linestyle='--')
    ax3.hlines(omega_bounds[1], t[0], t[-1], color='red', linestyle="--")
    ax3.set_ylim([-1 + omega_bounds[0], 1 + omega_bounds[1]])
    ax3.set_xlim([t[0], t[-1]])
    ax3.grid(True)

    # init and update function for the animation of simulation
    def init_sim():
        robot.set_offsets(robot_center[0, :])
        controlled_pt.set_offsets(configurations[0, :2])
        robot_clearance.set_center(robot_center[0, :])
        robot_clearance.set_radius(rho_cbf)
        ax_big.add_patch(robot_clearance)
        robot_label.set_position(robot_center[0])

        goal.set_offsets(targets[0, :2])
        goal_label.set_position(targets[0])
        
        for i in range(n_obstacles):
            obs_position = obstacles_position[0, i, :2]
            obstacles[i].set_offsets(obs_position)
            obstacles_clearance[i].set_center(obs_position)
            obstacles_clearance[i].set_radius(ds_cbf)
            ax_big.add_patch(obstacles_clearance[i])
            obstacles_label[i].set_position(obs_position)
        if n_obstacles > 0:
            return robot, robot_clearance, robot_label, goal, goal_label, \
                    obstacles, obstacles_clearance, obstacles_label
        else:
            return robot, robot_clearance, robot_label, goal, goal_label,

    def update_sim(frame):
        current_prediction = predictions[frame, :, :]
        current_target = targets[frame, :]

        wr_line.set_data(t[:frame + 1], inputs[:frame + 1, 0])
        wl_line.set_data(t[:frame + 1], inputs[:frame + 1, 1])
        v_line.set_data(t[:frame + 1], velocities[:frame + 1, 0])
        omega_line.set_data(t[:frame + 1], velocities[:frame + 1, 1])
        traj_line.set_data(configurations[:frame + 1, 0], configurations[:frame + 1, 1])
        pred_line.set_data(current_prediction[0, :], current_prediction[1, :])

        robot.set_offsets(robot_center[frame, :])
        controlled_pt.set_offsets(configurations[frame, :2])
        robot_clearance.set_center(robot_center[frame, :])
        robot_label.set_position(robot_center[frame, :])
        goal.set_offsets(current_target[:2])
        goal_label.set_position(current_target)
        for i in range(n_obstacles):
            obs_position = obstacles_position[frame, i , :2]
            obstacles[i].set_offsets(obs_position)
            obstacles_clearance[i].set_center(obs_position)
            obstacles_label[i].set_position(obs_position)

        if frame == shooting_nodes - 1:
            sim_animation.event_source.stop()

        if n_obstacles > 0:
            return robot, robot_clearance, robot_label, goal, goal_label, \
                    traj_line, pred_line, wr_line, wl_line, v_line, omega_line, \
                    obstacles, obstacles_clearance, obstacles_label
        else:
            return robot, robot_clearance, robot_label, goal, goal_label, \
                    traj_line, pred_line, wr_line, wl_line, v_line, omega_line,

    sim_animation = FuncAnimation(sim_fig, update_sim,
                                  frames=shooting_nodes,
                                  init_func=init_sim,
                                  blit=False,
                                  interval=1/frequency*1000,
                                  repeat=False)
    plt.tight_layout()
    # sim_animation.save(save_sim_path, writer='ffmpeg', fps=frequency, dpi=80)
    # print("Simulation saved")
    plt.show()
    return

def main():
    filename = rospy.get_param('/filename')
    plot_results(filename)
    return