import numpy as np
import rospy
import scipy.linalg

from acados_template import AcadosModel, AcadosOcp, AcadosOcpConstraints, AcadosOcpCost, AcadosOcpOptions, AcadosOcpSolver
from numpy.linalg import *

import casadi

from my_tiago_controller.Hparams import *
from my_tiago_controller.utils import *

class NMPC:
    def __init__(self,
                 hparams : Hparams):
        # Size of state and input:
        self.nq = 5
        self.nu = 2 # right and left wheel angular accelerations

        # Parameters:
        self.hparams = hparams

        # Number of control intervals:
        self.N = hparams.N_horizon

        # Horizon duration:
        dt = 1.0 / hparams.controller_frequency
        self.T = dt * self.N # [s]

        # Setup solver:
        self.acados_ocp_solver = self.__create_acados_ocp_solver(self.N,self.T)

    def init(self, x0: State):
        for k in range(self.N):
            self.acados_ocp_solver.set(k, 'x', x0.get_state())
            self.acados_ocp_solver.set(k, 'u', np.zeros(self.nu))
        self.acados_ocp_solver.set(self.N, 'x', x0.get_state())

    # Systems dynamics:
    def __f(self, x, u):
        xdot = casadi.SX.zeros(self.nq)
        xdot[self.hparams.x_idx] = self.__x_dot(x)
        xdot[self.hparams.y_idx] = self.__y_dot(x)
        xdot[self.hparams.theta_idx] = self.__theta_dot(x)
        xdot[self.hparams.v_idx] = self.__v_dot(u)
        xdot[self.hparams.omega_idx] = self.__omega_dot(u)
        return xdot

    def __x_dot(self, q):
        b = self.hparams.b
        theta = q[self.hparams.theta_idx]
        v = q[self.hparams.v_idx]
        omega = q[self.hparams.omega_idx]
        return v * casadi.cos(theta) - omega * b * casadi.sin(theta)

    def __y_dot(self, q):
        b = self.hparams.b
        theta = q[self.hparams.theta_idx]
        v = q[self.hparams.v_idx]
        omega = q[self.hparams.omega_idx]
        return v * casadi.sin(theta) + omega * b * casadi.cos(theta)
    
    def __theta_dot(self, q):
        return q[self.hparams.omega_idx]
    
    def __v_dot(self, u):
        alpha_r = u[self.hparams.r_wheel_idx]
        alpha_l = u[self.hparams.l_wheel_idx]
        wheel_radius = self.hparams.wheel_radius
        return wheel_radius * 0.5 * (alpha_r + alpha_l)
    
    def __omega_dot(self, u):
        alpha_r = u[self.hparams.r_wheel_idx]
        alpha_l = u[self.hparams.l_wheel_idx]
        wheel_radius = self.hparams.wheel_radius
        wheel_separation = self.hparams.wheel_separation
        return (wheel_radius / wheel_separation) * (alpha_r - alpha_l)
    
    def __h(self, q, p):
        x = q[self.hparams.x_idx]
        y = q[self.hparams.y_idx]
        theta = q[self.hparams.theta_idx]
        b = self.hparams.b

        n_actors = self.hparams.n_actors
        if n_actors > 0:
            h = casadi.SX.zeros(4 + n_actors)
        else:
            h = casadi.SX.zeros(4)

        # Consider the robot distance from the bounds [ubx, lbx, uby, lby]
        x_c = x - b * casadi.cos(theta)
        y_c = y - b * casadi.sin(theta)
        h[0] = self.hparams.x_upper_bound - x_c - self.hparams.rho_cbf
        h[1] = x_c - self.hparams.x_lower_bound - self.hparams.rho_cbf
        h[2] = self.hparams.y_upper_bound - y_c - self.hparams.rho_cbf
        h[3] = y_c - self.hparams.y_lower_bound - self.hparams.rho_cbf

        # Consider the robot distance from actors, if actors are present
        if n_actors > 0:
            distance_vectors = casadi.SX.zeros((n_actors, 2))
            cbf_radius = self.hparams.rho_cbf + self.hparams.ds_cbf
            for i in range(n_actors):
                distance_vectors[i, self.hparams.x_idx] = x_c - p[i*4 + self.hparams.x_idx]
                distance_vectors[i, self.hparams.y_idx] = y_c - p[i*4 + self.hparams.y_idx]
                h[i + 4] = distance_vectors[i, self.hparams.x_idx]**2 + \
                        distance_vectors[i, self.hparams.y_idx]**2 - \
                        cbf_radius**2

        return h

    def __h_dot(self, q, u, p):
        x = q[self.hparams.x_idx]
        y = q[self.hparams.y_idx]
        theta = q[self.hparams.theta_idx]
        v = q[self.hparams.v_idx]
        b = self.hparams.b

        # Set the coordinates of the robot center
        x_c = x - b * casadi.cos(theta)
        y_c = y - b * casadi.sin(theta)

        n_actors = self.hparams.n_actors
        if n_actors > 0:
            hdot = casadi.SX.zeros(4 + n_actors)
        else:
            hdot = casadi.SX.zeros(4)

        hdot[0] = - v * casadi.cos(theta)
        hdot[1] = v * casadi.cos(theta)
        hdot[2] = - v * casadi.sin(theta)
        hdot[3] = v * casadi.sin(theta)
        for i in range(n_actors):
            s1 = 2 * (x_c - p[i*4 + self.hparams.x_idx])
            s2 = 2 * (y_c - p[i*4 + self.hparams.y_idx])
            hdot[i + 4] = s1 * (v * casadi.cos(theta) - p[i*4 + 2 + self.hparams.x_idx]) + \
                          s2 * (v * casadi.sin(theta) - p[i*4 + 2 + self.hparams.y_idx])
            
        return hdot

    def __h_b(self, q, p):
        x = q[self.hparams.x_idx]
        y = q[self.hparams.y_idx]

        n_actors = self.hparams.n_actors
        if n_actors > 0:
            h = casadi.SX.zeros(4 + n_actors)
        else:
            h = casadi.SX.zeros(4)

        h[0] = self.hparams.x_upper_bound - x - self.hparams.rho_cbf
        h[1] = x - self.hparams.x_lower_bound - self.hparams.rho_cbf
        h[2] = self.hparams.y_upper_bound - y - self.hparams.rho_cbf
        h[3] = y - self.hparams.y_lower_bound - self.hparams.rho_cbf

        # Consider the robot distance from actors, if actors are present
        if n_actors > 0:
            distance_vectors = casadi.SX.zeros((n_actors, 2))
            cbf_radius = self.hparams.rho_cbf + self.hparams.ds_cbf
            for i in range(n_actors):
                distance_vectors[i, self.hparams.x_idx] = x - p[i*4 + self.hparams.x_idx]
                distance_vectors[i, self.hparams.y_idx] = y - p[i*4 + self.hparams.y_idx]
                h[i + 4] = distance_vectors[i, self.hparams.x_idx]**2 + \
                        distance_vectors[i, self.hparams.y_idx]**2 - \
                        cbf_radius**2
                
        return h

    def __h_dot_b(self, q, u, p):
        x = q[self.hparams.x_idx]
        y = q[self.hparams.y_idx]
        theta = q[self.hparams.theta_idx]
        v = q[self.hparams.v_idx]
        omega = q[self.hparams.omega_idx]
        b = self.hparams.b

        n_actors = self.hparams.n_actors
        if n_actors > 0:
            hdot = casadi.SX.zeros(4 + n_actors)
        else:
            hdot = casadi.SX.zeros(4)

        hdot[0] = - v * casadi.cos(theta) + omega * b * casadi.sin(theta)
        hdot[1] = v * casadi.cos(theta) - omega * b * casadi.sin(theta)
        hdot[2] = - v * casadi.sin(theta) - omega * b * casadi.cos(theta)
        hdot[3] = v * casadi.sin(theta) + omega * b * casadi.cos(theta)
        for i in range(n_actors):
            s1 = 2 * (x - p[i*4 + self.hparams.x_idx])
            s2 = 2 * (y - p[i*4 + self.hparams.y_idx])
            hdot[i + 4] = s1 * (v * casadi.cos(theta) - omega * b * casadi.sin(theta) - p[i*4 + 2 + self.hparams.x_idx]) + \
                          s2 * (v * casadi.sin(theta) + omega * b * casadi.cos(theta) - p[i*4 + 2 + self.hparams.y_idx])
            
        return hdot
    
    def __create_acados_model(self) -> AcadosModel:
        # Setup CasADi expressions:
        q = casadi.SX.sym('q', self.nq)
        qdot = casadi.SX.sym('qdot', self.nq)
        u = casadi.SX.sym('u', self.nu)
        p = casadi.SX.sym('p', self.hparams.n_actors * 4)
        f_expl = self.__f(q, u)
        f_impl = qdot - f_expl

        # Create acados model:
        acados_model = AcadosModel()
        acados_model.name = 'tiago_kinematic_model'

        # System dynamics:
        acados_model.f_impl_expr = f_impl
        acados_model.f_expl_expr = f_expl

        # CBF constraints:
        n_actors = self.hparams.n_actors
        gamma_mat = np.zeros((n_actors + 4, n_actors + 4))
        np.fill_diagonal(gamma_mat[:4, :4], self.hparams.gamma_bound)
        if n_actors > 0:
            np.fill_diagonal(gamma_mat[4:, 4:], self.hparams.gamma_actor)

        # if you consider robot center for the CBF constraints, uncomment next two lines
        # con_h_expr = self.__h_dot(q, u, p) + np.matmul(gamma_mat, self.__h(q, p))
        # acados_model.con_h_expr = con_h_expr

        # if you consider point B for the CBF constraints, uncomment next two lines
        con_h_expr = self.__h_dot_b(q, u, p) + np.matmul(gamma_mat, self.__h_b(q, p))
        acados_model.con_h_expr = con_h_expr
    


        # Variables and params:
        acados_model.x = q
        acados_model.xdot = qdot
        acados_model.u = u
        acados_model.p = p

        return acados_model
    
    def __create_acados_cost(self) -> AcadosOcpCost:
        acados_cost = AcadosOcpCost()

        # Set wheighting matrices
        Q_mat = np.diag([self.hparams.p_weight, self.hparams.p_weight, 0.0]) # [x, y, theta]
        R_mat = np.diag([self.hparams.v_weight, self.hparams.omega_weight]) # [v, omega]
        S_mat = np.diag([self.hparams.u_weight, self.hparams.u_weight]) # [alphar, alphal]

        acados_cost.cost_type   = 'LINEAR_LS'
        acados_cost.cost_type_e = 'LINEAR_LS'
        
        ny = self.nq + self.nu
        ny_e = self.nq

        acados_cost.W_e = scipy.linalg.block_diag(self.hparams.terminal_factor * Q_mat, R_mat)
        acados_cost.W = scipy.linalg.block_diag(Q_mat, R_mat, S_mat)

        Vx = np.zeros((ny, self.nq))
        Vx[:self.nq, :self.nq] = np.eye(self.nq)
        acados_cost.Vx = Vx

        Vu = np.zeros((ny, self.nu))
        Vu[self.nq : ny, 0 : self.nu] = np.eye(self.nu)
        acados_cost.Vu = Vu

        acados_cost.Vx_e = np.eye(ny_e)

        acados_cost.yref = np.zeros((ny,))
        acados_cost.yref_e = np.zeros((ny_e,))
        
        return acados_cost
    
    def __create_acados_constraints(self) -> AcadosOcpConstraints:

        acados_constraints = AcadosOcpConstraints()

        # Linear inequality constraints on the state:
        acados_constraints.idxbx = np.array([self.hparams.v_idx, self.hparams.omega_idx])
        acados_constraints.lbx = np.array([self.hparams.driving_vel_min, self.hparams.steering_vel_max_neg])
        acados_constraints.ubx = np.array([self.hparams.driving_vel_max, self.hparams.steering_vel_max])
        acados_constraints.x0 = np.zeros(self.nq)

        # Linear inequality constraints on the inputs:
        acados_constraints.idxbu = np.array([self.hparams.r_wheel_idx, self.hparams.l_wheel_idx])
        acados_constraints.lbu = np.array([self.hparams.alpha_min, self.hparams.alpha_min])
        acados_constraints.ubu = np.array([self.hparams.alpha_max, self.hparams.alpha_max])

        # Linear constraints on wheel velocities and driving/steering acceleration
        # expressed in terms of state and input
        C_mat = np.zeros((4, self.nq))
        C_mat[:2, 3] = (1 / self.hparams.wheel_radius)
        C_mat[:2, 4] = self.hparams.wheel_separation / (2 * self.hparams.wheel_radius) * np.array([1, -1])
        D_mat = np.zeros((4, self.nu))
        D_mat[2, :] = self.hparams.wheel_radius * 0.5
        D_mat[3, :] = (self.hparams.wheel_radius / self.hparams.wheel_separation) * np.array([1, -1])
        acados_constraints.D = D_mat
        acados_constraints.C = C_mat
        acados_constraints.lg = np.array([self.hparams.w_max_neg,
                                          self.hparams.w_max_neg,
                                          self.hparams.driving_acc_min,
                                          self.hparams.steering_acc_max_neg])
        acados_constraints.ug = np.array([self.hparams.w_max,
                                          self.hparams.w_max,
                                          self.hparams.driving_acc_max,
                                          self.hparams.steering_acc_max])

        # Nonlinear constraints (CBFs) (for both actors and configuration bounds):
        if self.hparams.n_actors > 0:
            acados_constraints.lh = np.zeros(self.hparams.n_actors + 4)
            acados_constraints.uh = 10000 * np.ones(self.hparams.n_actors + 4)
        else:
            acados_constraints.lh = np.zeros(4)
            acados_constraints.uh = 10000 * np.ones(4) 

        return acados_constraints
    
    def __create_acados_solver_options(self, T) -> AcadosOcpOptions:
        acados_solver_options = AcadosOcpOptions()
        acados_solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        acados_solver_options.hpipm_mode = 'SPEED'
        acados_solver_options.hessian_approx = 'GAUSS_NEWTON'
        acados_solver_options.integrator_type = 'ERK'
        acados_solver_options.print_level = 0
        acados_solver_options.nlp_solver_type = 'SQP_RTI'
        acados_solver_options.tf = T

        return acados_solver_options
    
    def __create_acados_ocp(self, N, T) -> AcadosOcp:
        acados_ocp = AcadosOcp()
        acados_ocp.model = self.__create_acados_model()
        acados_ocp.dims.N = N
        acados_ocp.parameter_values = np.zeros((self.hparams.n_actors * 4,))
        acados_ocp.cost = self.__create_acados_cost()
        acados_ocp.constraints = self.__create_acados_constraints()
        acados_ocp.solver_options = self.__create_acados_solver_options(T)
        
        return acados_ocp
    
    def __create_acados_ocp_solver(self, N, T, use_cython=False) -> AcadosOcpSolver:
        acados_ocp = self.__create_acados_ocp(N, T)
        if use_cython:
            AcadosOcpSolver.generate(acados_ocp, json_file='acados_ocp_nlp.json')
            AcadosOcpSolver.build(acados_ocp.code_export_directory, with_cython=True)
            return AcadosOcpSolver.create_cython_solver('acados_ocp_nlp.json')
        else:
            return AcadosOcpSolver(acados_ocp)

    def update(
            self,
            state: State,
            q_ref: np.array,
            u_ref: np.array,
            crowd_motion_prediction : CrowdMotionPrediction
            ):
        # Set parameters
        for k in range(self.N):
            self.acados_ocp_solver.set(k, 'y_ref', np.concatenate((q_ref[:, k], u_ref[:, k])))
            actors_state = np.zeros((self.hparams.n_actors * 4))
            for j in range(self.hparams.n_actors):
                actors_state[j*4 + 0] = crowd_motion_prediction.motion_predictions[j].positions[k].x
                actors_state[j*4 + 1] = crowd_motion_prediction.motion_predictions[j].positions[k].y
                actors_state[j*4 + 2] = crowd_motion_prediction.motion_predictions[j].velocities[k].x
                actors_state[j*4 + 3] = crowd_motion_prediction.motion_predictions[j].velocities[k].y
            self.acados_ocp_solver.set(k, 'p', actors_state)
        self.acados_ocp_solver.set(self.N, 'y_ref', q_ref[:, self.N])

        # Solve NLP
        self.u0 = self.acados_ocp_solver.solve_for_x0(state.get_state())

    def get_command(self):
        return self.u0
