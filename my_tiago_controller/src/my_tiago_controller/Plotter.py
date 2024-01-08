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
    save_plot1_path = os.path.join(save_plots_dir, filename + '_plot1.png')
    save_plot2_path = os.path.join(save_plots_dir, filename + '_plot2.png')
    save_sim_path = os.path.join(save_sim_dir, filename + '_simulation.mp4')

    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            data = json.load(file)
    else:
        raise Exception(
            f"Specified file not found"
        )

    # Setup all the data
    states = np.array(data['states'])
    configurations = states[:, :3]
    robot_center = np.empty((configurations.shape[0], 2))
    b = data['offset_b']
    for i in range(configurations.shape[0]):
        robot_center[i, 0] = configurations[i, 0] - b * math.cos(configurations[i, 2])
        robot_center[i, 1] = configurations[i, 1] - b * math.sin(configurations[i, 2])
    
    driving_velocities = states[:, 3]
    steering_velocities = states[:, 4]
    robot_predictions = np.array(data['robot_predictions'])
    inputs = np.array(data['wheels_accelerations'])
    wheels_velocities = np.array(data['wheels_velocities'])
    targets = np.array(data['targets'])

    x_bounds = np.array(data['x_bounds'])
    y_bounds = np.array(data['y_bounds'])
    input_bounds = np.array(data['input_bounds'])
    v_bounds = np.array(data['v_bounds'])
    omega_bounds = np.array(data['omega_bounds'])
    wheels_vel_bounds = np.array(data['wheels_vel_bounds'])

    n_actors = data['n_actors']
    n_clusters = data['n_clusters']
    if n_actors > 0:
        actors_predictions = np.array(data['actors_predictions'])
        actors_groundtruth = np.array(data['actors_gt'])
        
    N_horizon = data['N_horizon']
    rho_cbf = data['rho_cbf']
    ds_cbf = data['ds_cbf']
    frequency = data['frequency']
    shooting_nodes = inputs.shape[0]
    t = inputs[:, 2]

    # Figure plot1
    plot1_fig, axs1 = plt.subplots(3, 2, figsize=(16, 8))

    axs1[0, 0].plot(t, wheels_velocities[:, 0], label='$\omega_r$')
    axs1[0, 0].plot(t, wheels_velocities[:, 1], label='$\omega_l$')
    axs1[1, 0].plot(t, driving_velocities[:], label='$v$')
    axs1[2, 0].plot(t, steering_velocities[:], label='$\omega$')
    axs1[0, 1].plot(t, configurations[:, 0], label='$x$')
    axs1[0, 1].plot(t, targets[:, 0], label='$x_g$')
    axs1[1, 1].plot(t, configurations[:, 1], label='$y$')
    axs1[1, 1].plot(t, targets[:, 1], label="$y_g$")
    axs1[2, 1].plot(t, configurations[:, 2], label='$\theta$')

    axs1[0, 0].set_title('wheels angular velocity')
    axs1[0, 0].set_xlabel("$t \quad [s]$")
    axs1[0, 0].set_ylabel('$[rad/s]$')
    axs1[0, 0].legend(loc='upper left')
    axs1[0, 0].hlines(wheels_vel_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs1[0, 0].hlines(wheels_vel_bounds[1], t[0], t[-1], color='red', linestyle="--")
    axs1[0, 0].set_ylim([-1 + wheels_vel_bounds[0], 1 + wheels_vel_bounds[1]])
    axs1[0, 0].set_xlim([t[0], t[-1]])
    axs1[0, 0].grid(True)

    axs1[1, 0].set_title('TIAGo driving velocity')
    axs1[1, 0].set_xlabel("$t \quad [s]$")
    axs1[1, 0].set_ylabel('$[m/s]$')
    axs1[1, 0].hlines(v_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs1[1, 0].hlines(v_bounds[1], t[0], t[-1], color='red', linestyle="--")
    axs1[1, 0].set_ylim([-1 + v_bounds[0], 1 + v_bounds[1]])
    axs1[1, 0].set_xlim([t[0], t[-1]])
    axs1[1, 0].grid(True)

    axs1[2, 0].set_title('TIAGo steering velocity')
    axs1[2, 0].set_xlabel("$t \quad [s]$")
    axs1[2, 0].set_ylabel('$[rad/s]$')
    axs1[2, 0].hlines(omega_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs1[2, 0].hlines(omega_bounds[1], t[0], t[-1], color='red', linestyle="--")
    axs1[2, 0].set_ylim([-1 + omega_bounds[0], 1 + omega_bounds[1]])
    axs1[2, 0].set_xlim([t[0], t[-1]])
    axs1[2, 0].grid(True)

    axs1[0, 1].set_title('$x-position$')
    axs1[0, 1].set_xlabel('$t \quad [s]$')
    axs1[0, 1].set_ylabel('$[m]$')
    axs1[0, 1].legend(loc='upper left')
    axs1[0, 1].hlines(x_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs1[0, 1].hlines(x_bounds[1], t[0], t[-1], color='red', linestyle='--')
    axs1[0, 1].set_ylim([-1 + x_bounds[0], 1 + x_bounds[1]])
    axs1[0, 1].set_xlim([t[0], t[-1]])
    axs1[0, 1].grid(True)

    axs1[1, 1].set_title('$y-position$')
    axs1[1, 1].set_xlabel('$t \quad [s]$')
    axs1[1, 1].set_ylabel('$[m]$')
    axs1[1, 1].legend(loc='upper left')
    axs1[1, 1].hlines(y_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs1[1, 1].hlines(y_bounds[1], t[0], t[-1], color='red', linestyle='--')
    axs1[1, 1].set_ylim([-1 + y_bounds[0], 1 + y_bounds[1]])
    axs1[1, 1].set_xlim([t[0], t[-1]])
    axs1[1, 1].grid(True)

    axs1[2, 1].set_title('TIAGo orientation')
    axs1[2, 1].set_xlabel('$t \quad [s]$')
    axs1[2, 1].set_ylabel('$[rad]$')
    axs1[2, 1].set_ylim([-1 + np.min(configurations[:, 2]), 1 + np.max(configurations[:, 2])])
    axs1[2, 1].set_xlim([t[0], t[-1]])
    axs1[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_plot1_path)
    plt.show()

    # Figure plot2
    plot2_fig, axs2 = plt.subplots(3, 1, figsize=(16, 8))
    axs2[0].plot(t, targets[:, 0] - configurations[:, 0], label='$e_x$')
    axs2[0].plot(t, targets[:, 1] - configurations[:, 1], label='$e_y$')
    axs2[1].plot(t, wheels_velocities[:, 0], label='$\omega_r$')
    axs2[1].plot(t, wheels_velocities[:, 1], label='$\omega_l$')
    axs2[2].plot(t, inputs[:, 0], label='$\\alpha_r$')
    axs2[2].plot(t, inputs[:, 1], label='$\\alpha_l$')

    axs2[0].set_title('position errors')
    axs2[0].set_xlabel("$t \quad [s]$")
    axs2[0].set_ylabel('$[m]$')
    axs2[0].legend(loc='upper left')
    axs2[0].set_ylim([-1 + np.min([x_bounds[0], y_bounds[0]]), 1 + np.max([x_bounds[1], y_bounds[1]])])
    axs2[0].set_xlim([t[0], t[-1]])
    axs2[0].grid(True)

    axs2[1].set_title('wheels angular velocity')
    axs2[1].set_xlabel("$t \quad [s]$")
    axs2[1].set_ylabel('$[rad/s]$')
    axs2[1].legend(loc='upper left')
    axs2[1].hlines(wheels_vel_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs2[1].hlines(wheels_vel_bounds[1], t[0], t[-1], color='red', linestyle="--")
    axs2[1].set_ylim([-1 + wheels_vel_bounds[0], 1 + wheels_vel_bounds[1]])
    axs2[1].set_xlim([t[0], t[-1]])
    axs2[1].grid(True)

    axs2[2].set_title('wheels accelerations')
    axs2[2].set_xlabel("$t \quad [s]$")
    axs2[2].set_ylabel('$[rad/s^2]$')
    axs2[2].legend(loc='upper left')
    axs2[2].hlines(input_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs2[2].hlines(input_bounds[1], t[0], t[-1], color='red', linestyle="--")
    axs2[2].set_ylim([-1 + input_bounds[0], 1 + input_bounds[1]])
    axs2[2].set_xlim([t[0], t[-1]])
    axs2[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_plot2_path)
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
    robot_clearance = Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='blue')
    goal = ax_big.scatter([], [], s=80.0, marker='*', label='goal', color='magenta', alpha=0.7)
    goal_label = ax_big.text(np.nan, np.nan, goal.get_label(), fontsize=8, ha='left', va='bottom')
    if n_actors > 0:
        actors = []
        actors_label = []
        actors_clearance = []
        actors_pred_line = []
        actors_gt = []
        actors_gt_label = []
        actors_gt_clearance = []
        
        for i in range(n_clusters):
            actors.append(ax_big.scatter([], [], marker='o', label='actor{}'.format(i+1), color='red', alpha=0.7))
            actors_clearance.append(Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='red'))
            actors_label.append(ax_big.text(np.nan, np.nan, actors[i].get_label(), fontsize=8, ha='left', va='bottom'))
            actor_pred_line, = ax_big.plot([], [], color='orange', label='actor prediction')
            actors_pred_line.append(actor_pred_line)
    
        for i in range(n_actors):
            actors_gt.append(ax_big.scatter([], [], marker='o', label='actor{}'.format(i+1), color='cyan', alpha=0.7))
            actors_gt_clearance.append(Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='cyan'))
            actors_gt_label.append(ax_big.text(np.nan, np.nan, actors_gt[i].get_label(), fontsize=8, ha='left', va='bottom'))
    
    traj_line, = ax_big.plot([], [], color='blue', label='trajectory')
    robot_pred_line, = ax_big.plot([], [], color='green', label='prediction')
    ex_line, = ax1.plot([], [], label='$e_x$')
    ey_line, = ax1.plot([], [], label='$e_y$')
    wr_line, = ax2.plot([], [], label='$\omega_r$')
    wl_line, = ax2.plot([], [], label='$\omega_l$')
    alphar_line, = ax3.plot([], [], label='$\\alpha_r$')
    alphal_line, = ax3.plot([], [], label='$\\alpha_l$')

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

    ax1.set_title('position errors')
    ax1.set_xlabel("$t \quad [s]$")
    ax1.set_ylabel('$[m]$')
    ax1.legend(loc='upper left')
    ax1.set_ylim([-1 + np.min([x_bounds[0], y_bounds[0]]), 1 + np.max([x_bounds[1], y_bounds[1]])])
    ax1.set_xlim([t[0], t[-1]])
    ax1.grid(True)

    ax2.set_title('wheels velocities')
    ax2.set_xlabel("$t \quad [s]$")
    ax2.set_ylabel('$[rad/s]$')
    ax2.legend(loc='upper left')
    ax2.hlines(wheels_vel_bounds[0], t[0], t[-1], color='red', linestyle='--')
    ax2.hlines(wheels_vel_bounds[1], t[0], t[-1], color='red', linestyle="--")
    ax2.set_ylim([-1 + wheels_vel_bounds[0], 1 + wheels_vel_bounds[1]])
    ax2.set_xlim([t[0], t[-1]])
    ax2.grid(True)

    ax3.set_title('wheels accelerations')
    ax3.set_xlabel("$t \quad [s]$")
    ax3.set_ylabel('$[rad/s^2]$')
    ax3.legend(loc='upper left')
    ax3.hlines(input_bounds[0], t[0], t[-1], color='red', linestyle='--')
    ax3.hlines(input_bounds[1], t[0], t[-1], color='red', linestyle="--")
    ax3.set_ylim([-1 + input_bounds[0], 1 + input_bounds[1]])
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

        if n_actors > 0:        
            for i in range(n_clusters):
                actor_position = actors_predictions[0, i, :, 0]
                actors[i].set_offsets(actor_position)
                actors_clearance[i].set_center(actor_position)
                actors_clearance[i].set_radius(ds_cbf)
                ax_big.add_patch(actors_clearance[i])
                actors_label[i].set_position(actor_position)

        for i in range(n_actors):
            actor_gt_position = actors_groundtruth[0, i, :]
            actors_gt[i].set_offsets(actor_gt_position)
            actors_gt_clearance[i].set_center(actor_gt_position)
            actors_gt_clearance[i].set_radius(ds_cbf)
            ax_big.add_patch(actors_gt_clearance[i])
            actors_gt_label[i].set_position(actor_gt_position)

        if n_actors > 0:
            return robot, robot_clearance, robot_label, goal, goal_label, \
                    actors, actors_clearance, actors_label, \
                    actors_gt, actors_gt_clearance, actors_gt_label
        else:
            return robot, robot_clearance, robot_label, goal, goal_label,

    def update_sim(frame):
        robot_prediction = robot_predictions[frame, :, :]
        current_target = targets[frame, :]

        ex_line.set_data(t[:frame + 1], targets[:frame + 1, 0] - configurations[:frame + 1, 0])
        ey_line.set_data(t[:frame + 1], targets[:frame + 1, 1] - configurations[:frame + 1, 1])
        wr_line.set_data(t[:frame + 1], wheels_velocities[:frame + 1, 0])
        wl_line.set_data(t[:frame + 1], wheels_velocities[:frame + 1, 1])
        alphar_line.set_data(t[:frame + 1], inputs[:frame + 1, 0])
        alphal_line.set_data(t[:frame + 1], inputs[:frame + 1, 1])
        traj_line.set_data(configurations[:frame + 1, 0], configurations[:frame + 1, 1])
        robot_pred_line.set_data(robot_prediction[0, :], robot_prediction[1, :])

        robot.set_offsets(robot_center[frame, :])
        controlled_pt.set_offsets(configurations[frame, :2])
        robot_clearance.set_center(robot_center[frame, :])
        robot_label.set_position(robot_center[frame, :])
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

        for i in range(n_actors):
            actor_gt_position = actors_groundtruth[frame, i, :]
            actors_gt[i].set_offsets(actor_gt_position)
            actors_gt_clearance[i].set_center(actor_gt_position)
            actors_gt_label[i].set_position(actor_gt_position)

        if frame == shooting_nodes - 1:
            sim_animation.event_source.stop()

        if n_actors > 0:
            return robot, robot_clearance, robot_label, goal, goal_label, \
                    ex_line, ey_line, wr_line, wl_line, alphar_line, alphal_line, traj_line, robot_pred_line, \
                    actors, actors_clearance, actors_label, \
                    actors_gt, actors_gt_clearance, actors_gt_label
        else:
            return robot, robot_clearance, robot_label, goal, goal_label, \
                    ex_line, ey_line, wr_line, wl_line, alphar_line, alphal_line, traj_line, robot_pred_line

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