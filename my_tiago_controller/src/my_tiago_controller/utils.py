import matplotlib.pyplot as plt
import numpy as np
import os

def plot_robot(
    shooting_nodes,
    u_max,
    U,
    X_sim,
    save=False,
    save_dir=None
):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: array with shape (N_sim, nu)
        X_sim: array with shape (N_sim+1, nx)

    """

    if save:
        if save_dir==None:
            raise Exception(
                f"Directory to save plots not specified"
            )
        else:
            plots_filename = 'sim_plots.png'
            plots_path = os.path.join(save_dir, plots_filename)

    nx = X_sim.shape[1]
    nu = U.shape[1]
    t = shooting_nodes

    control_labels = ["$w_r$", "$w_l$"]
    for i in range(nu):
        plt.subplot(nx + nu, 1, i+1)
        # line, = plt.step(t, np.append([U[0]], U))
        # line, = plt.plot(t, U[:, 0], label='U')
        (line,) = plt.step(t, np.append([U[0, i]], U[:, i]))
        # (line,) = plt.step(t, np.append([U[0, 0]], U[:, 0]))
        line.set_color("r")
        # plt.title('closed-loop simulation')
        plt.ylabel(control_labels[i])
        plt.xlabel("$t$")
        if u_max[i] is not None:
            plt.hlines(u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            plt.hlines(-u_max[i], t[0], t[-1], linestyles="dashed", alpha=0.7)
            plt.ylim([-1.2 * u_max[i], 1.2 * u_max[i]])
        plt.grid()

    states_labels = ["$x$", "$y$", "$theta$"]
    for i in range(nx):
        plt.subplot(nx + nu, 1, i + nu+1)
        (line,) = plt.plot(t, X_sim[:, i])
        plt.ylabel(states_labels[i])
        plt.xlabel("$t$")
        plt.grid()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    plt.savefig(plots_path)
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