import numpy as np

class Hparams:
    target_point = np.array([5.0,3.0]) # [m]
    wheel_radius = 0.0985 # [m]
    wheel_separation = 0.4044 # [m]
    # input velocity limit for both left and right wheel
    w_max = 10 #[rad/s]
    w_max_neg = -w_max
    # configuration limits (only for cartesian position)
    lower_bound = -10 # [m]
    upper_bound = 10  # [m]
    # State indices:
    x_idx = 0
    y_idx = 1
    theta_idx = 2
    # Control input indices
    wr_idx = 0
    wl_idx = 1

