import math
import numpy as np

from my_tiago_controller.Hparams import *
 
class KinematicModel:
    nq = 3 # q = (x, y, theta)
    nu = 2 # u = (w_r, w_l)

    def __init__(self):
        self.hparams = Hparams()

    def __call__(self, q, u):
        theta = q[self.hparams.theta_idx]
        w_r = u[self.hparams.wr_idx]
        w_l = u[self.hparams.wl_idx]
        wheel_radius = self.hparams.wheel_radius
        wheel_separation = self.hparams.wheel_separation

        xdot = (wheel_radius/2)*math.cos(theta)*(w_r+w_l)
        ydot = (wheel_radius/2)*math.sin(theta)*(w_r+w_l)
        thetadot = (wheel_radius/wheel_separation)*(w_r-w_l)
        qdot = np.array([xdot, ydot, thetadot])
        return qdot