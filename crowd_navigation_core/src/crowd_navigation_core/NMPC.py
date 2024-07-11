import numpy as np

from acados_template import AcadosModel, AcadosOcp, AcadosOcpConstraints, AcadosOcpCost, AcadosOcpOptions, AcadosOcpSolver
from numpy.linalg import *

import casadi

from crowd_navigation_core.Hparams import *
from crowd_navigation_core.utils import *

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
        self.dt = self.hparams.dt
        self.T = self.dt * self.N # [s]

        self.max_vertexes = self.hparams.max_vertexes
        self.areas = self.hparams.areas
        self.n_filters = self.hparams.n_filters
        
        # Number of parameters:
        self.np = 4 * self.n_filters + 3 * self.max_vertexes

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

    def Euler(self, f, x0, u, dt):
        return x0 + f(x0,u)*dt

    def RK4(self, f, x0, u ,dt):
        k1 = f(x0, u)
        k2 = f(x0 + k1 * dt / 2.0, u)
        k3 = f(x0 + k2 * dt / 2.0, u)
        k4 = f(x0 + k3 * dt, u)
        yf = x0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return yf

    def integrate(self, f, x0, u, integration_method='RK4'):
        dt = self.dt
        if integration_method == 'RK4':
            return self.RK4(f, x0, u, dt)
        else:
            return self.Euler(f, x0, u, dt)

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

    def __create_acados_model(self) -> AcadosModel:
        # Setup CasADi expressions:
        q = casadi.SX.sym('q', self.nq)
        qdot = casadi.SX.sym('qdot', self.nq)
        u = casadi.SX.sym('u', self.nu)
        r = casadi.SX.sym('r', self.nq + self.nu)
        r_e = casadi.SX.sym('r_e', self.nq)
        p = casadi.SX.sym('p', self.np)
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
        
        acados_model.con_h_expr = self.cbf_constraint(q, u, p)

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
        acados_constraints.lh = np.zeros(self.n_filters + self.max_vertexes)
        acados_constraints.uh = self.hparams.unbounded * np.ones(self.n_filters + self.max_vertexes)

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
        acados_ocp.parameter_values = np.zeros(self.np)
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
        
    def compute_agent_cbf(self,
                          robot_state,
                          agent_position):
        robot_position = casadi.vertcat(robot_state[self.hparams.x_idx],
                                        robot_state[self.hparams.y_idx])
        distance = casadi.norm_2(robot_position - agent_position)
        h = distance**2 - (self.hparams.rho_cbf + self.hparams.ds_cbf)**2
        return h
    
    def compute_area_cbf(self,
                         robot_state,
                         coefs):
        robot_position = casadi.vertcat(robot_state[self.hparams.x_idx],
                                        robot_state[self.hparams.y_idx])
        a = coefs[0]
        b = coefs[1]
        c = coefs[2]
        h = - a * robot_position[0] - b * robot_position[1] + c - self.hparams.rho_cbf
        return h
    
    def cbf_constraint(self, x, u, p):
        h_constraints = []
        next_x = self.integrate(self.__f, x, u)
        for i in range(self.n_filters):
            current_agent_position = p[4*i:4*i+2]
            next_agent_position = p[4*i+2:4*i+4]
            current_h = self.compute_agent_cbf(x, current_agent_position)
            next_h = self.compute_agent_cbf(next_x, next_agent_position)
            cbf_condition = next_h + (self.hparams.gamma_agent - 1) * current_h
            h_constraints.append(cbf_condition)

        area_p = p[4*self.n_filters:]
        for i in range(self.max_vertexes):
            coefs = area_p[3*i:3*i+3]
            current_h = self.compute_area_cbf(x, coefs)
            next_h = self.compute_area_cbf(next_x, coefs)
            cbf_condition = next_h + (self.hparams.gamma_area - 1) * current_h
            h_constraints.append(cbf_condition)
        return casadi.vertcat(*h_constraints)

    def predict_trajectory(self, position, velocity):
        predicted_trajectory = np.zeros((self.N+1, 2))
        dt = self.dt
        state = np.array([position.x,
                          position.y,
                          velocity.x,
                          velocity.y])
        for i in range(self.N+1):
            time = dt * (i + 1)
            predicted_trajectory[i] = predict_next_position(state, time)

        return predicted_trajectory

    def update(self,
               state: State,
               q_ref: np.array,
               u_ref: np.array,
               area_index,
               crowd_motion_predictions):
        # Set parameters
        self.predicted_agent_trajectories = []
        for motion_prediction in crowd_motion_predictions:
            self.predicted_agent_trajectories.append(self.predict_trajectory(motion_prediction.position,
                                                                             motion_prediction.velocity))
        for k in range(self.N):
            self.acados_ocp_solver.set(k, 'y_ref', np.concatenate((q_ref[:, k], u_ref[:, k])))
            parameters = []
            for agent_trajectory in self.predicted_agent_trajectories:
                current_agent_position = agent_trajectory[k]
                next_agent_position = agent_trajectory[k+1]
                parameters.extend(current_agent_position)
                parameters.extend(next_agent_position)
            for a,b,c in zip(self.hparams.a_coefs[area_index], self.hparams.b_coefs[area_index], self.hparams.c_coefs[area_index]):
                parameters.extend([a, b, c])
            self.acados_ocp_solver.set(k, 'p', np.array(parameters))
        self.acados_ocp_solver.set(self.N, 'y_ref', q_ref[:, self.N])

        # Solve NLP
        self.u0 = self.acados_ocp_solver.solve_for_x0(state.get_state())

    def get_control_input(self):
        return self.u0

    def get_predicted_agent_trajectories(self):
        return self.predicted_agent_trajectories