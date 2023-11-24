import numpy as np

class Hparams:
    target_point = np.array([1.4 , 0.5]) # [m]
    wheel_radius = 0.0985
    wheel_separation = 0.4044
    w_max = 10 # [rad/s]
    w_max_neg = -w_max
    lbx = -10
    ubx = 10
    # State indices:
    x_idx = 0
    y_idx = 1
    theta_idx = 2
    # Control input indices
    wr_idx = 0
    wl_idx = 1

    # input velocity limit for both left and right wheel
    w_max = 10 #[rad/s]
    w_max_neg = -w_max