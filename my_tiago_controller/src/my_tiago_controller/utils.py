import numpy as np

class Configuration:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.theta)
    
    def get_q(self):
        return np.array([self.x, self.y, self.theta])

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
    
def linear_trajectory(xi, xf, n_steps):
    """
    Generate a linear trajectory between two 2D points.

    Parameters:
    - xi: Initial point (tuple or array-like)
    - xf: Final point (tuple or array-like)
    - n_steps: Number of steps for the trajectory (integer)

    Returns:
    - trajectory: 2D array containing the trajectory points
    """

    # Check if the dimensions of xi and xf match
    assert xi.shape == xf.shape == (2,), "Initial and final points must be 2D."

    # Initialize the trajectory array
    trajectory = np.zeros((n_steps, 2))

    # Generate linear trajectory
    for i in range(n_steps):
        alpha = i / (n_steps - 1)  # Interpolation parameter
        trajectory[i] = (1 - alpha) * xi + alpha * xf

    return trajectory
