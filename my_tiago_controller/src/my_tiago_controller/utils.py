import json
import matplotlib.pyplot as plt
import numpy as np
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
    save_path = os.path.join(save_dir, filename + '.png')

    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            data = json.load(file)
    else:
        raise Exception(
            f"Specified file not found"
        )

    configurations = np.array(data['configurations'])
    inputs = np.array(data['inputs'])
    predictions = np.array(data['predictions'])
    control_bounds = np.array(data['control_bounds'])
    x_bounds = np.array(data['x_bounds'])
    y_bounds = np.array(data['y_bounds'])
    shooting_nodes = inputs.shape[0]
    t = inputs[:, 2]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))

    line1, = ax1.plot([], [])
    line2, = ax2.plot([], [])
    scatter3 = ax3.scatter([], [], marker='o', facecolors='none', edgecolors='blue')
    line3, = ax3.plot([], [], color='blue')
    line4, = ax3.plot([], [], color='green')

    ax1.set_title('right wheel angular velocity')
    ax1.set_xlabel("$t$")
    ax1.set_ylabel('$w_r$')
    ax1.hlines(control_bounds[0], t[0], t[-1], color='red', linestyle='--')
    ax1.hlines(control_bounds[1], t[0], t[-1], color='red', linestyle="--")
    ax1.set_ylim([1.2 * control_bounds[0], 1.2 * control_bounds[1]])

    ax2.set_title('left wheel angular velocity')
    ax2.set_xlabel("$t$")
    ax2.set_ylabel('$w_l$')
    ax2.hlines(control_bounds[0], t[0], t[-1], color='red', linestyle='--')
    ax2.hlines(control_bounds[1], t[0], t[-1], color='red', linestyle="--")
    ax2.set_ylim([1.2 * control_bounds[0], 1.2 * control_bounds[1]])

    ax3.set_title('Trajectory')
    ax3.set_xlabel("$x$")
    ax3.set_ylabel('$y$')
    ax3.axhline(y_bounds[0], color='red', linestyle='--')
    ax3.axhline(y_bounds[1], color='red', linestyle="--")
    ax3.axvline(x_bounds[0], color='red', linestyle='--')
    ax3.axvline(x_bounds[1], color='red', linestyle='--')
    ax3.set_ylim([1.2 * y_bounds[0], 1.2 * y_bounds[1]])
    ax3.set_xlim([1.2 * x_bounds[0], 1.2 * x_bounds[1]])

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        scatter3.set_offsets([])
        line3.set_data([], [])
        line4.set_data([], [])
        return line1, line2, scatter3, line3, line4
    
    def update(frame):
        current_inputs = inputs[:frame + 1, :]
        current_configurations = configurations[:frame +1, :]
        current_prediction = predictions[frame, :, :]
        current_time = t[:frame + 1]

        line1.set_data(current_time, current_inputs[:, 0])
        line2.set_data(current_time, current_inputs[:, 1])
        scatter3.set_offsets(current_configurations[frame, :2])
        line3.set_data(current_configurations[:, 0], current_configurations[:, 1])
        line4.set_data(current_prediction[0, :], current_prediction[1, :])

        if frame == shooting_nodes - 1:
            animation.event_source.stop()
            plt.savefig(save_path)
        return line1, line2, scatter3, line3, line4
    
    animation = FuncAnimation(fig, update, frames=shooting_nodes, init_func=init, blit=True, interval=10, repeat=False)
    
    plt.tight_layout()
    plt.show()

def Euler(f, x0, u, dt):
    return x0 + f(x0,u)*dt

def RK4(f, x0, u ,dt):
    k1 = f(x0, u)
    k2 = f(x0 + k1 * dt / 2.0, u)
    k3 = f(x0 + k2 * dt / 2.0, u)
    k4 = f(x0 + k3 * dt, u)
    yf = x0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return yf

def integrate(f, x0, u, dt, integration_method='RK4'):
    if integration_method == 'RK4':
        return RK4(f, x0, u, dt)
    else:
        return Euler(f, x0, u, dt)