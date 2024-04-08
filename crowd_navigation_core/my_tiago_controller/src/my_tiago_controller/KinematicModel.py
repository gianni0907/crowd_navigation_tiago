import casadi
from my_tiago_controller.Hparams import *

class KinematicModel:
    nq = 5
    nu = 2

    def __init__(self):
        self.hparams = Hparams()

    # q = (x, y, theta, v, omega)^T
    # u = (alpha_r, alpha_l)^T
    def __call__(self, q, u):
        x = q[self.hparams.x_idx]
        y = q[self.hparams.y_idx]
        theta = q[self.hparams.theta_idx]
        v = q[self.hparams.v_idx]
        omega = q[self.hparams.omega_idx]
        alpha_r = u[self.hparams.r_wheel_idx]
        alpha_l = u[self.hparams.l_wheel_idx]

        b = self.hparams.b
        wheel_radius = self.hparams.wheel_radius
        wheel_separation = self.hparams.wheel_separation

        xdot = v * casadi.cos(theta) - omega * b * casadi.sin(theta)
        ydot = v * casadi.sin(theta) + omega * b * casadi.cos(theta)
        thetadot = omega
        vdot = wheel_radius * 0.5 * (alpha_r + alpha_l)
        omegadot = (wheel_radius / wheel_separation) * (alpha_r - alpha_l)

        qdot = np.array([xdot, ydot, thetadot, vdot, omegadot])
        return qdot