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
    plots_savedir = '/tmp/crowd_navigation_tiago/plots'
    if not os.path.exists(plots_savedir):
        os.makedirs(plots_savedir)

    # Specify saving animations directory
    animation_savedir = '/tmp/crowd_navigation_tiago/animations'
    if not os.path.exists(animation_savedir):
        os.makedirs(animation_savedir)

    log_controller = os.path.join(log_dir, filename + '_controller.json')
    log_predictor = os.path.join(log_dir, filename + '_predictor.json')
    profiles1_savepath = os.path.join(plots_savedir, filename + '_profiles1.png')
    profiles2_savepath = os.path.join(plots_savedir, filename + '_profiles2.png')
    time_savepath = os.path.join(plots_savedir, filename + '_time.png')
    scans_savepath = os.path.join(animation_savedir, filename + '_scans.mp4')
    world_savepath = os.path.join(animation_savedir, filename + '_world.mp4')

    # Open the controller log file
    if os.path.exists(log_controller):
        with open(log_controller, 'r') as file:
            controller_dict = json.load(file)
    else:
        raise Exception(
            f"Specified file not found"
        )

    # Extract the controller data
    iteration_time = np.array(controller_dict['cpu_time'])
    states = np.array(controller_dict['states'])
    configurations = states[:, :3]
    robot_center = np.empty((configurations.shape[0], 2))
    b = controller_dict['offset_b']
    for i in range(configurations.shape[0]):
        robot_center[i, 0] = configurations[i, 0] - b * math.cos(configurations[i, 2])
        robot_center[i, 1] = configurations[i, 1] - b * math.sin(configurations[i, 2])
    
    driving_velocities = states[:, 3]
    steering_velocities = states[:, 4]
    robot_predictions = np.array(controller_dict['robot_predictions'])
    inputs = np.array(controller_dict['wheels_accelerations'])
    wheels_velocities = np.array(controller_dict['wheels_velocities'])
    velocities = np.array(controller_dict['velocities'])
    targets = np.array(controller_dict['targets'])
    errors = targets[:, :2] - configurations[:, :2]
    n_edges = controller_dict['n_edges']
    boundary_vertexes = np.array(controller_dict['boundary_vertexes'])
    input_bounds = np.array(controller_dict['input_bounds'])
    v_bounds = np.array(controller_dict['v_bounds'])
    omega_bounds = np.array(controller_dict['omega_bounds'])
    wheels_vel_bounds = np.array(controller_dict['wheels_vel_bounds'])

    n_actors = controller_dict['n_actors']
    n_clusters = controller_dict['n_clusters']
    simulation = controller_dict['simulation']
    if n_actors > 0:
        fake_sensing = controller_dict['fake_sensing']
        actors_predictions = np.array(controller_dict['actors_predictions'])
        if simulation and not fake_sensing:
            actors_groundtruth = np.array(controller_dict['actors_gt'])
        
    rho_cbf = controller_dict['rho_cbf']
    ds_cbf = controller_dict['ds_cbf']
    frequency = controller_dict['frequency']
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
        predictor_time = np.array(predictor_dict['cpu_time'])
        laser_scans = predictor_dict['laser_scans']
        actors_position = np.array(predictor_dict['actors_position'])
        robot_states = np.array(predictor_dict['robot_states'])
        robot_config = robot_states[:, :3]

    # Figure elapsed time per iteration (controller and predictor if prediction module is present)
    fig, axs = plt.subplots(2, 1, figsize=(16, 8))
    
    axs[0].step(t, iteration_time[:, 0])
    axs[0].set_title('Elapsed time per controller iteration')
    axs[0].set_xlabel('$t \quad [s]$')
    axs[0].set_ylabel('$iteration \quad time \quad [s]$')
    axs[0].hlines(1 / frequency, t[0], t[-1], color='red', linestyle='--')
    axs[0].set_xlim([t[0], t[-1]])
    axs[0].grid(True)

    if n_actors > 0:
        axs[1].step(predictor_time[:, 1], predictor_time[:, 0])
        axs[1].set_title('Elapsed time per predictor iteration')
        axs[1].set_xlabel('$t \quad [s]$')
        axs[1].set_ylabel('$iteration \quad time \quad [s]$')
        axs[1].hlines(1 / frequency, predictor_time[0, 1], predictor_time[-1, 1], color='red', linestyle='--')
        axs[1].set_xlim(predictor_time[0, 1], predictor_time[-1, 1])
        axs[1].grid(True)

    fig.tight_layout()
    fig.savefig(time_savepath)

    # Figure profiles1
    profiles1_fig, axs1 = plt.subplots(3, 2, figsize=(16, 8))

    axs1[0, 0].plot(t, wheels_velocities[:, 0], label='$\omega_r$')
    axs1[0, 0].plot(t, wheels_velocities[:, 1], label='$\omega_l$')
    axs1[1, 0].plot(t, driving_velocities[:], label='$v$')
    axs1[1, 0].plot(t, velocities[:, 0], label='$v_{cmd}$')
    axs1[2, 0].plot(t, steering_velocities[:], label='$\omega$')
    axs1[2, 0].plot(t, velocities[:, 1], label='$\omega_{cmd}$')
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
    axs1[1, 0].legend(loc='upper left')
    axs1[1, 0].hlines(v_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs1[1, 0].hlines(v_bounds[1], t[0], t[-1], color='red', linestyle="--")
    axs1[1, 0].set_ylim([-1 + v_bounds[0], 1 + v_bounds[1]])
    axs1[1, 0].set_xlim([t[0], t[-1]])
    axs1[1, 0].grid(True)

    axs1[2, 0].set_title('TIAGo steering velocity')
    axs1[2, 0].set_xlabel("$t \quad [s]$")
    axs1[2, 0].set_ylabel('$[rad/s]$')
    axs1[2, 0].legend(loc='upper left')
    axs1[2, 0].hlines(omega_bounds[0], t[0], t[-1], color='red', linestyle='--')
    axs1[2, 0].hlines(omega_bounds[1], t[0], t[-1], color='red', linestyle="--")
    axs1[2, 0].set_ylim([-1 + omega_bounds[0], 1 + omega_bounds[1]])
    axs1[2, 0].set_xlim([t[0], t[-1]])
    axs1[2, 0].grid(True)

    axs1[0, 1].set_title('$x-position$')
    axs1[0, 1].set_xlabel('$t \quad [s]$')
    axs1[0, 1].set_ylabel('$[m]$')
    axs1[0, 1].legend(loc='upper left')
    axs1[0, 1].set_xlim([t[0], t[-1]])
    axs1[0, 1].grid(True)

    axs1[1, 1].set_title('$y-position$')
    axs1[1, 1].set_xlabel('$t \quad [s]$')
    axs1[1, 1].set_ylabel('$[m]$')
    axs1[1, 1].legend(loc='upper left')
    axs1[1, 1].set_xlim([t[0], t[-1]])
    axs1[1, 1].grid(True)

    axs1[2, 1].set_title('TIAGo orientation')
    axs1[2, 1].set_xlabel('$t \quad [s]$')
    axs1[2, 1].set_ylabel('$[rad]$')
    axs1[2, 1].set_ylim([-1 + np.min(configurations[:, 2]), 1 + np.max(configurations[:, 2])])
    axs1[2, 1].set_xlim([t[0], t[-1]])
    axs1[2, 1].grid(True)
    
    profiles1_fig.tight_layout()
    profiles1_fig.savefig(profiles1_savepath)

    # Figure profiles2
    plot2_fig, axs2 = plt.subplots(3, 1, figsize=(16, 8))
    axs2[0].plot(t, errors[:, 0], label='$e_x$')
    axs2[0].plot(t, errors[:, 1], label='$e_y$')
    axs2[1].plot(t, wheels_velocities[:, 0], label='$\omega_r$')
    axs2[1].plot(t, wheels_velocities[:, 1], label='$\omega_l$')
    axs2[2].plot(t, inputs[:, 0], label='$\\alpha_r$')
    axs2[2].plot(t, inputs[:, 1], label='$\\alpha_l$')

    axs2[0].set_title('position errors')
    axs2[0].set_xlabel("$t \quad [s]$")
    axs2[0].set_ylabel('$[m]$')
    axs2[0].legend(loc='upper left')
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

    plot2_fig.tight_layout()
    plot2_fig.savefig(profiles2_savepath)

    plt.show()

    # Figure to plot world animation
    world_fig = plt.figure(figsize=(16, 8))
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
            actors.append(ax_big.scatter([], [], marker='.', label='fsm{}'.format(i+1), color='red', alpha=0.7))
            actors_clearance.append(Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='red'))
            actors_label.append(ax_big.text(np.nan, np.nan, actors[i].get_label(), fontsize=8, ha='left', va='bottom'))
            actor_pred_line, = ax_big.plot([], [], color='orange', label='actor prediction')
            actors_pred_line.append(actor_pred_line)

        if simulation and not fake_sensing:
            for i in range(n_actors):
                actors_gt.append(ax_big.scatter([], [], marker='.', label='actor{}'.format(i+1), color='cyan', alpha=0.7))
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
    boundary_line = []
    for i in range(n_edges - 1):
        x_values = [boundary_vertexes[i, 0], boundary_vertexes [i + 1, 0]]
        y_values = [boundary_vertexes[i, 1], boundary_vertexes [i + 1, 1]]
        line, = ax_big.plot(x_values, y_values, color='red', linestyle='--')
        boundary_line.append(line)
    x_values = [boundary_vertexes[n_edges - 1, 0], boundary_vertexes [0, 0]]
    y_values = [boundary_vertexes[n_edges - 1, 1], boundary_vertexes [0, 1]]
    line, = ax_big.plot(x_values, y_values, color='red', linestyle='--')
    boundary_line.append(line)

    ax_big.set_title('TIAGo World')
    ax_big.set_xlabel("$x \quad [m]$")
    ax_big.set_ylabel('$y \quad [m]$')
    ax_big.set_aspect('equal', adjustable='box')
    ax_big.grid(True)

    ax1.set_title('position errors')
    ax1.set_xlabel("$t \quad [s]$")
    ax1.set_ylabel('$[m]$')
    ax1.legend(loc='upper left')
    ax1.set_ylim([-1 + np.min(errors), 1 + np.max(errors)])
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

    # init and update function for the world animation
    def init_world():
        robot.set_offsets(robot_center[0, :])
        controlled_pt.set_offsets(configurations[0, :2])
        robot_clearance.set_center(configurations[0, :2])
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

            if simulation and not fake_sensing:
                for i in range(n_actors):
                    actor_gt_position = actors_groundtruth[0, i, :]
                    actors_gt[i].set_offsets(actor_gt_position)
                    actors_gt_clearance[i].set_center(actor_gt_position)
                    actors_gt_clearance[i].set_radius(ds_cbf)
                    ax_big.add_patch(actors_gt_clearance[i])
                    actors_gt_label[i].set_position(actor_gt_position)

                return robot, robot_clearance, robot_label, goal, goal_label, \
                       actors, actors_clearance, actors_label, \
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
        robot_clearance.set_center(configurations[frame, :2])
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

            if simulation and not fake_sensing:
                for i in range(n_actors):
                    actor_gt_position = actors_groundtruth[frame, i, :]
                    actors_gt[i].set_offsets(actor_gt_position)
                    actors_gt_clearance[i].set_center(actor_gt_position)
                    actors_gt_label[i].set_position(actor_gt_position)

                return robot, robot_clearance, robot_label, goal, goal_label, \
                       ex_line, ey_line, wr_line, wl_line, alphar_line, alphal_line, traj_line, robot_pred_line, \
                       actors, actors_clearance, actors_label, actors_pred_line, \
                       actors_gt, actors_gt_clearance, actors_gt_label
            else:
                return robot, robot_clearance, robot_label, goal, goal_label, \
                       ex_line, ey_line, wr_line, wl_line, alphar_line, alphal_line, traj_line, robot_pred_line, \
                       actors, actors_clearance, actors_label, actors_pred_line
        else:
            return robot, robot_clearance, robot_label, goal, goal_label, \
                   ex_line, ey_line, wr_line, wl_line, alphar_line, alphal_line, traj_line, robot_pred_line

    world_animation = FuncAnimation(world_fig, update_world,
                                    frames=shooting_nodes,
                                    init_func=init_world,
                                    blit=False,
                                    interval=1/frequency*100,
                                    repeat=False)
    world_fig.tight_layout()
    # world_animation.save(world_savepath, writer='ffmpeg', fps=frequency, dpi=80)
    # print("World animation saved")
    
    plt.show()

    # Figure to plot scans animation
    if n_actors > 0 and not fake_sensing:
        scans_fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(1,1)
        ax = plt.subplot(gs[0, 0])

        robot = ax.scatter([], [], s=100.0, marker='o',label='TIAGo', facecolors='none', edgecolors='blue')
        controlled_pt = ax.scatter([], [], marker='.', color='blue')
        robot_label = ax.text(np.nan, np.nan, robot.get_label(), fontsize=8, ha='left', va='bottom')
        robot_clearance = Circle(np.zeros(1), np.zeros(1), facecolor='none', edgecolor='blue')
        scans, = ax.plot([], [], color='magenta', marker='.', linestyle='', label='scans')
        core_points = []
        for i in range(n_clusters):
            point, = ax.plot([], [], color='orange', marker='x', linestyle='', label='actor')
            core_points.append(point)

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

        # init and update function for the world animation
        def init_scans():
            robot.set_offsets(robot_center[0, :])
            controlled_pt.set_offsets(robot_config[0, :2])
            robot_clearance.set_center(robot_config[0, :2])
            robot_clearance.set_radius(rho_cbf)
            ax.add_patch(robot_clearance)
            robot_label.set_position(robot_center[0])
        
            return robot, robot_clearance, robot_label

        def update_scans(frame):
            if frame == shooting_nodes - 1:
                scans_animation.event_source.stop()

            robot.set_offsets(robot_center[frame, :])
            controlled_pt.set_offsets(robot_config[frame, :2])
            robot_clearance.set_center(robot_config[frame, :2])
            robot_label.set_position(robot_center[frame, :])
            current_scans = np.array(laser_scans[frame])
            if current_scans.shape[0] > 0:
                scans.set_data(current_scans[:, 0], current_scans[:, 1])
                for i in range(n_clusters):
                    actor_position = actors_position[frame, i, :]
                    if any(coord != 0.0 for coord in actor_position):
                        core_points[i].set_data(actor_position[0], actor_position[1])
            else:
                scans.set_data([], [])
                for i in range(n_clusters):
                    core_points[i].set_data([], [])

            return robot, robot_clearance, robot_label, scans, core_points

        scans_animation = FuncAnimation(scans_fig, update_scans,
                                        frames=shooting_nodes,
                                        init_func=init_scans,
                                        blit=False,
                                        interval=1/frequency*500,
                                        repeat=False)
        scans_fig.tight_layout()
        # scans_animation.save(scans_savepath, writer='ffmpeg', fps=frequency, dpi=80)
        # print("Scans animation saved")
        
        plt.show()

def main():
    filename = rospy.get_param('/filename')
    plot_results(filename)