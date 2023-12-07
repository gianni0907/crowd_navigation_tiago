import numpy as np
import rospy
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from  matplotlib.animation import FuncAnimation

def plot_results(filename=None):
    # Specify logging directory
    log_dir = '/tmp/crowd_navigation_tiago/data'
    if not os.path.exists(log_dir):
        raise Exception(
           f"Specified directory not found"
        )
    # Specify saving plots directory
    save_dir = '/tmp/crowd_navigation_tiago/plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_path = os.path.join(log_dir, filename + '.json')
    save_vel_path = os.path.join(save_dir, filename + '_velocities.png')
    save_sim_path = os.path.join(save_dir, filename + '_simulation.png')

    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            data = json.load(file)
    else:
        raise Exception(
            f"Specified file not found"
        )

    # Setup all the data
    configurations = np.array(data['configurations'])
    inputs = np.array(data['inputs'])
    predictions = np.array(data['predictions'])
    velocities = np.array(data['velocities'])
    control_bounds = np.array(data['control_bounds'])
    x_bounds = np.array(data['x_bounds'])
    y_bounds = np.array(data['y_bounds'])
    v_bounds = np.array(data['v_bounds'])
    omega_bounds = np.array(data['omega_bounds'])
    obstacles_position = np.array(data['obstacles_position'])
    print(velocities.shape)
    print(inputs.shape)
    print(configurations.shape)
    rho_cbf = np.array(data['rho_cbf'])
    ds_cbf = np.array(data['ds_cbf'])
    shooting_nodes = inputs.shape[0]
    n_obstacles = obstacles_position.shape[0]
    t = inputs[:, 2]
    print(t[-1])    
    # Initialize figure to plot velocities
    vel_fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    wr_line, = ax1.plot([], [], label='$\omega_r$')
    wl_line, = ax1.plot([], [], label='$\omega_l$')
    v_line, = ax2.plot([], [], label='$v$')
    omega_line, = ax3.plot([], [], label='$\omega$')

    ax1.set_title('wheels angular velocity')
    ax1.set_xlabel("$[t]$")
    ax1.set_ylabel('$[rad/s]$')
    ax1.legend(loc='upper right')
    ax1.hlines(control_bounds[0], t[0], t[-1], color='red', linestyle='--')
    ax1.hlines(control_bounds[1], t[0], t[-1], color='red', linestyle="--")
    ax1.set_ylim([1.2 * control_bounds[0], 1.2 * control_bounds[1]])

    ax2.set_title('TIAGo driving velocity')
    ax2.set_xlabel("$t$")
    ax2.set_ylabel('$[m/s]$')
    ax2.hlines(v_bounds[0], t[0], t[-1], color='red', linestyle='--')
    ax2.hlines(v_bounds[1], t[0], t[-1], color='red', linestyle="--")
    ax2.set_ylim([1.2 * v_bounds[0], 1.2 * v_bounds[1]])

    ax3.set_title('TIAGo steering velocity')
    ax3.set_xlabel("$t$")
    ax3.set_ylabel('$[rad/s]$')
    ax3.hlines(omega_bounds[0], t[0], t[-1], color='red', linestyle='--')
    ax3.hlines(omega_bounds[1], t[0], t[-1], color='red', linestyle="--")
    ax3.set_ylim([1.2 * omega_bounds[0], 1.2 * omega_bounds[1]])

    # init and update function for the animation of velocity plots
    def init_vel():
        wr_line.set_data([], [])
        wl_line.set_data([], [])
        v_line.set_data([], [])
        omega_line.set_data([], [])
        return wr_line, wl_line, v_line, omega_line
    
    def update_vel(frame):
        current_inputs = inputs[:frame + 1, :]
        current_velocities = velocities[:frame + 1, :]
        current_time = t[:frame + 1]

        wr_line.set_data(current_time, current_inputs[:, 0])
        wl_line.set_data(current_time, current_inputs[:, 1])
        v_line.set_data(current_time, current_velocities[:, 0])
        omega_line.set_data(current_time, current_velocities[:, 1])

        if frame == shooting_nodes - 1:
            vel_animation.event_source.stop()
            plt.savefig(save_vel_path)
        return wr_line, wl_line, v_line, omega_line

    sim_fig = plt.figure()
    robot = plt.scatter([], [], marker='o', label='TIAGo', facecolors='none', edgecolors='blue')
    robot_clearance = Circle(configurations[0, :2], rho_cbf, facecolor='none', edgecolor='blue')
    obstacles = []
    obstacles_clearance = []
    for i in range(n_obstacles):
        obstacles.append(plt.scatter([], [], marker='o', label='obstacle{}'.format(i+1),color='red'))
    traj_line, = plt.plot([], [], color='blue', label='trajectory')
    pred_line, = plt.plot([], [], color='green', label='prediction')

    plt.suptitle('Simulation')
    plt.xlabel("$x \quad [m]$")
    plt.ylabel('$y \quad [m]$')
    plt.axhline(y_bounds[0], color='red', linestyle='--')
    plt.axhline(y_bounds[1], color='red', linestyle="--")
    plt.axvline(x_bounds[0], color='red', linestyle='--')
    plt.axvline(x_bounds[1], color='red', linestyle='--')
    plt.ylim([1.2 * y_bounds[0], 1.2 * y_bounds[1]])
    plt.xlim([1.2 * x_bounds[0], 1.2 * x_bounds[1]])
    plt.gca().set_aspect('equal', adjustable='box')

    # init and update function for the animation of simulation
    def init_sim():
        robot.set_offsets(configurations[0, :2])
        plt.gca().add_patch(robot_clearance)
        for i in range(n_obstacles):
            obs_position = obstacles_position[i, :]
            obstacles[i].set_offsets(obs_position)
            circle = Circle(obs_position, ds_cbf, facecolor='none', edgecolor='red')
            plt.gca().add_patch(circle)
            obstacles_clearance.append(circle)
        traj_line.set_data([], [])
        pred_line.set_data([], [])
        return robot, robot_clearance, traj_line, pred_line
    
    def update_sim(frame):
        current_configurations = configurations[:frame +1, :]
        current_prediction = predictions[frame, :, :]

        robot.set_offsets(current_configurations[frame, :2])
        robot_clearance.set_center(current_configurations[frame, :2])
        traj_line.set_data(current_configurations[:, 0], current_configurations[:, 1])
        pred_line.set_data(current_prediction[0, :], current_prediction[1, :])

        if frame == shooting_nodes - 1:
            sim_animation.event_source.stop()
            plt.savefig(save_sim_path)
        return robot, robot_clearance, traj_line, pred_line
    
    vel_animation = FuncAnimation(vel_fig, update_vel,
                                  frames=shooting_nodes,
                                  init_func=init_vel,
                                  blit=True,
                                  interval=10,
                                  repeat=False)
    
    sim_animation = FuncAnimation(sim_fig, update_sim,
                                  frames=shooting_nodes,
                                  init_func=init_sim,
                                  blit=True,
                                  interval=10,
                                  repeat=False)

    
    vel_fig.tight_layout()
    plt.show()

def main():
    filename = rospy.get_param('/filename')
    plot_results(filename)