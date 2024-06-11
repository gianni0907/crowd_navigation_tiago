import numpy as np

from acados_template import AcadosModel, AcadosOcp, AcadosOcpConstraints, AcadosOcpCost, AcadosOcpOptions, AcadosOcpSolver
from numpy.linalg import *

import casadi

from crowd_navigation_core.Hparams import *
from crowd_navigation_core.utils import *
from crowd_navigation_core.KinematicModel import *

class NMPC:
    def __init__(self,
                 hparams : Hparams):
        # Size of state and input:
        self.nq = 5
        self.nu = 2 # right and left wheel angular accelerations

        # Size of agents state:
        self.agent_state_size = 4

        # Parameters:
        self.hparams = hparams

        # Number of control intervals:
        self.N = hparams.N_horizon

        # Horizon duration:
        self.dt = self.hparams.dt
        self.T = self.dt * self.N # [s]

        # Setup kinematic model
        self.kinematic_model = KinematicModel()

        self.n_edges = self.hparams.n_points
        self.vertexes = self.hparams.vertexes
        self.normals = self.hparams.normals
        self.n_filters = self.hparams.n_filters

        # Setup solver:
        self.acados_ocp_solver = self.__create_acados_ocp_solver(self.N,self.T)

    def init(self, x0: State):
        unbounded = self.hparams.unbounded
        lg_0 = np.array([-unbounded,
                         -unbounded,
                         -unbounded,
                         -unbounded,
                         self.hparams.alpha_min,
                         self.hparams.alpha_min,
                         self.hparams.driving_acc_min,
                         self.hparams.steering_acc_max_neg])
        
        ug_0 = np.array([unbounded,
                         unbounded,
                         unbounded,
                         unbounded,
                         self.hparams.alpha_max,
                         self.hparams.alpha_max,
                         self.hparams.driving_acc_max,
                         self.hparams.steering_acc_max])
        
        self.acados_ocp_solver.constraints_set(0, 'lg', lg_0)
        self.acados_ocp_solver.constraints_set(0, 'ug', ug_0)

        lg = np.array([self.hparams.driving_vel_min,
                       self.hparams.steering_vel_max_neg,
                       self.hparams.w_max_neg,
                       self.hparams.w_max_neg,
                       self.hparams.alpha_min,
                       self.hparams.alpha_min,
                       self.hparams.driving_acc_min,
                       self.hparams.steering_acc_max_neg])
        
        ug = np.array([self.hparams.driving_vel_max,
                       self.hparams.steering_vel_max,
                       self.hparams.w_max,
                       self.hparams.w_max,
                       self.hparams.alpha_max,
                       self.hparams.alpha_max,
                       self.hparams.driving_acc_max,
                       self.hparams.steering_acc_max])
        
        for k in range(1, self.N):
            self.acados_ocp_solver.constraints_set(k, 'lg', lg)
            self.acados_ocp_solver.constraints_set(k, 'ug', ug)

        for k in range(self.N):
            self.acados_ocp_solver.set(k, 'x', x0.get_state())
            self.acados_ocp_solver.set(k, 'u', np.zeros(self.nu))
        self.acados_ocp_solver.set(self.N, 'x', x0.get_state())

    def __Euler(self, f, x0, u, dt):
        return x0 + f(x0,u)*dt

    def __RK4(self, f, x0, u ,dt):
        k1 = f(x0, u)
        k2 = f(x0 + k1 * dt / 2.0, u)
        k3 = f(x0 + k2 * dt / 2.0, u)
        k4 = f(x0 + k3 * dt, u)
        yf = x0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return yf

    def __integrate(self, f, x0, u, integration_method='RK4'):
        dt = self.dt
        if integration_method == 'RK4':
            return self.__RK4(f, x0, u, dt)
        else:
            return self.__Euler(f, x0, u, dt)

    def __next_agent_state(self, state):
        dt = self.dt
        F = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        F_complete= np.kron(np.eye(self.n_filters), F)
        next_state = np.matmul(F_complete, state)
        return next_state

    def __psi_expr(self, r):
        # Given a variable s, es = s - s_des 
        ex = r[self.hparams.x_idx]
        ey = r[self.hparams.y_idx]
        # Note: desired theta is zero, thus etheta=theta
        # In this application, the same holds also for v, omega and inputs
        theta = r[self.hparams.theta_idx] 
        v = r[self.hparams.v_idx] 
        omega = r[self.hparams.omega_idx]
        u = r[self.nq:]

        # Set weights
        p_weight = self.hparams.p_weight
        v_weight = self.hparams.v_weight
        omega_weight = self.hparams.omega_weight
        u_weight = self.hparams.u_weight
        h_weight = self.hparams.h_weight

        # Define the running cost function
        cost = p_weight * ex ** 2 + p_weight * ey ** 2 + \
               v_weight * v ** 2 + omega_weight * omega ** 2 + \
               u_weight * casadi.sumsqr(u) + \
               h_weight * (ex * casadi.cos(theta) + ey * casadi.sin(theta))
        return cost

    def __psi_expr_e(self, r):
        # Given a variable s, es = s - s_des 
        ex = r[self.hparams.x_idx]
        ey = r[self.hparams.y_idx]
        # Note: desired theta is zero, thus etheta=theta
        # In this application, the same holds also for v and omega
        theta = r[self.hparams.theta_idx] 
        v = r[self.hparams.v_idx] 
        omega = r[self.hparams.omega_idx]

        # Set weights
        p_weight = self.hparams.p_weight * self.hparams.terminal_factor_p
        v_weight = self.hparams.v_weight * self.hparams.terminal_factor_v
        omega_weight = self.hparams.omega_weight
        h_weight = self.hparams.h_weight

        # Define the terminal cost function
        cost = p_weight * ex ** 2 + p_weight * ey ** 2 + \
               v_weight * v ** 2 + omega_weight * omega ** 2 + \
               h_weight * (ex * casadi.cos(theta) + ey * casadi.sin(theta))
        return cost

    def __y_expr(self, q, u):
        return casadi.vertcat(q, u)
    
    def __y_expr_e(self, q):
        return q

    # Systems dynamics:
    def __f(self, q, u):
        xdot = casadi.SX.zeros(self.nq)
        xdot[self.hparams.x_idx] = self.__x_dot(q)
        xdot[self.hparams.y_idx] = self.__y_dot(q)
        xdot[self.hparams.theta_idx] = self.__theta_dot(q)
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

        h = casadi.SX.zeros(self.n_edges + self.n_filters)

        # Define the safe set wrt the configuration bounds
        robot_position = np.array([x, y])
        for i in range(self.n_edges):
            vertex = self.vertexes[i]
            h[i] = np.dot(self.normals[i], robot_position - vertex) - self.hparams.rho_cbf
        
        # Consider the robot distance from agents, if agents are present
        if self.n_filters > 0:
            cbf_radius = self.hparams.rho_cbf + self.hparams.ds_cbf
            for i in range(self.n_filters):
                sx = x - p[i*self.agent_state_size + self.hparams.x_idx]
                sy = y - p[i*self.agent_state_size + self.hparams.y_idx]
                h[i + self.n_edges] = sx**2 + sy**2 - cbf_radius**2
        return h

    def __create_acados_model(self) -> AcadosModel:
        # Setup CasADi expressions:
        q = casadi.SX.sym('q', self.nq)
        qdot = casadi.SX.sym('qdot', self.nq)
        u = casadi.SX.sym('u', self.nu)
        r = casadi.SX.sym('r', self.nq + self.nu)
        r_e = casadi.SX.sym('r_e', self.nq)
        p = casadi.SX.sym('p', self.n_filters * self.agent_state_size)
        f_expl = self.__f(q, u)
        f_impl = qdot - f_expl

        # Create acados model:
        acados_model = AcadosModel()
        acados_model.name = 'tiago_extended_model'

        # Cost function:
        acados_model.cost_psi_expr = self.__psi_expr(r)
        acados_model.cost_psi_expr_0 = self.__psi_expr(r)
        acados_model.cost_psi_expr_e = self.__psi_expr_e(r_e)
        acados_model.cost_r_in_psi_expr = r
        acados_model.cost_r_in_psi_expr_0 = r
        acados_model.cost_r_in_psi_expr_e = r_e
        acados_model.cost_y_expr = self.__y_expr(q, u)
        acados_model.cost_y_expr_0 = self.__y_expr(q, u)
        acados_model.cost_y_expr_e = self.__y_expr_e(q)
        
        # System dynamics:
        acados_model.f_impl_expr = f_impl
        acados_model.f_expl_expr = f_expl

        # CBF constraints:
        gamma_mat = np.zeros((self.n_edges + self.n_filters, self.n_edges + self.n_filters))
        id_mat = np.eye(self.n_edges + self.n_filters)
        np.fill_diagonal(gamma_mat[:self.n_edges, :self.n_edges], self.hparams.gamma_bound)
        if self.n_filters > 0:
            np.fill_diagonal(gamma_mat[self.n_edges:, self.n_edges:], self.hparams.gamma_agent)

        h_k = self.__h(q, p)
        
        q_k1 = self.__integrate(self.kinematic_model, q, u)
        if self.n_filters > 0:
            p_k1 = self.__next_agent_state(p)
        else:
            p_k1 = p # just for consistency, it is not used in case of 0 agents
        h_k1 = self.__h(q_k1, p_k1)
        con_h_expr = h_k1 + np.matmul(gamma_mat - id_mat, h_k)
        
        acados_model.con_h_expr = con_h_expr
        acados_model.con_h_expr_0 = con_h_expr

        # Variables and params:
        acados_model.x = q
        acados_model.xdot = qdot
        acados_model.u = u
        acados_model.p = p

        return acados_model
    
    def __create_acados_cost(self) -> AcadosOcpCost:
        acados_cost = AcadosOcpCost()

        acados_cost.cost_type   = 'CONVEX_OVER_NONLINEAR'
        acados_cost.cost_type_e = 'CONVEX_OVER_NONLINEAR'
        
        ny = self.nq + self.nu
        ny_e = self.nq

        acados_cost.yref = np.zeros((ny,))
        acados_cost.yref_e = np.zeros((ny_e,))
        
        return acados_cost
    
    def __create_acados_constraints(self) -> AcadosOcpConstraints:

        acados_constraints = AcadosOcpConstraints()

        # Initial constraint
        acados_constraints.x0 = np.zeros(self.nq)

        # Linear inequality constraints on the state and input:
        acados_constraints.lg = np.zeros(8)
        acados_constraints.ug = np.zeros(8)
        acados_constraints.lg_e = np.array([self.hparams.driving_vel_min,
                                            self.hparams.steering_vel_max_neg,
                                            self.hparams.w_max_neg,
                                            self.hparams.w_max_neg])
        acados_constraints.ug_e = np.array([self.hparams.driving_vel_max,
                                            self.hparams.steering_vel_max,
                                            self.hparams.w_max,
                                            self.hparams.w_max])

        C_mat = np.zeros((8, self.nq))
        C_mat[:2, 3:5] = np.eye(2)
        C_mat[2:4, 3] = (1 / self.hparams.wheel_radius)
        C_mat[2:4, 4] = self.hparams.wheel_separation / (2 * self.hparams.wheel_radius) * np.array([1, -1])
        acados_constraints.C = C_mat
        acados_constraints.C_e = C_mat[:4, :]

        D_mat = np.zeros((8, self.nu))
        D_mat[4:6] = np.eye(2)
        D_mat[6, :] = self.hparams.wheel_radius * 0.5
        D_mat[7, :] = (self.hparams.wheel_radius / self.hparams.wheel_separation) * np.array([1, -1])
        acados_constraints.D = D_mat

        # Nonlinear constraints (CBFs) (for both agents and configuration bounds):
        acados_constraints.lh = np.zeros(self.n_edges + self.n_filters)
        acados_constraints.uh = self.hparams.unbounded * np.ones(self.n_edges + self.n_filters)
        acados_constraints.lh_0 = - self.hparams.unbounded * np.ones(self.n_edges + self.n_filters)
        acados_constraints.uh_0 = acados_constraints.uh

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
        acados_ocp.parameter_values = np.zeros((self.n_filters * self.agent_state_size,))
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
        velocities = [Velocity() for _ in range(self.n_filters)]
        for j, motion_prediction in enumerate(crowd_motion_prediction.motion_predictions):
            velocities[j].x = (motion_prediction.positions[1].x - motion_prediction.positions[0].x) / self.dt
            velocities[j].y = (motion_prediction.positions[1].y - motion_prediction.positions[0].y) / self.dt
        for k in range(self.N):
            self.acados_ocp_solver.set(k, 'y_ref', np.concatenate((q_ref[:, k], u_ref[:, k])))
            agents_state = np.zeros((self.hparams.n_filters * self.agent_state_size))
            for j, motion_prediction in enumerate(crowd_motion_prediction.motion_predictions):
                position = crowd_motion_prediction.motion_predictions[j].positions[k]
                agents_state[j*self.agent_state_size + 0] = position.x
                agents_state[j*self.agent_state_size + 1] = position.y
                agents_state[j*self.agent_state_size + 2] = velocities[j].x
                agents_state[j*self.agent_state_size + 3] = velocities[j].y
            self.acados_ocp_solver.set(k, 'p', agents_state)
        self.acados_ocp_solver.set(self.N, 'y_ref', q_ref[:, self.N])

        # Solve NLP
        self.u0 = self.acados_ocp_solver.solve_for_x0(state.get_state())

    def get_control_input(self):
        return self.u0
