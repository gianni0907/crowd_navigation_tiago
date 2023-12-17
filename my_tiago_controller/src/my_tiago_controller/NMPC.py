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
        self.nq = 3
        self.nu = 2 # right and left wheel angular velocities

        # Parameters:
        self.hparams = hparams

        # Number of control intervals:
        self.N = hparams.N_horizon

        # Horizon duration:
        dt = 1.0 / hparams.controller_frequency
        self.T = dt * self.N # [s]

        # Setup solver:
        self.acados_ocp_solver = self.__create_acados_ocp_solver(self.N,self.T)

        # Variables for Analysis of required time
        self.tmp_time = 0.0
        self.max_time = 0.0
        self.idx_time = 0.0

    def init(self, x0: Configuration):
        # lbx = np.array([self.hparams.x_lower_bound, self.hparams.y_lower_bound])
        # ubx = np.array([self.hparams.x_upper_bound, self.hparams.y_upper_bound])   
        if self.hparams.n_obstacles > 0:
            lh = np.zeros(self.hparams.n_obstacles + 4)
            uh = 10000*np.ones(self.hparams.n_obstacles + 4)
        else:
            lh = np.zeros(4)
            uh = 10000*np.ones(4)

        for k in range(self.N):
            self.acados_ocp_solver.set(k, 'x', np.array(x0.get_q()))
            self.acados_ocp_solver.set(k, 'u', np.zeros(self.nu))
            self.acados_ocp_solver.constraints_set(k, 'lh', lh)
            self.acados_ocp_solver.constraints_set(k, 'uh', uh)
        self.acados_ocp_solver.set(self.N, 'x', np.array(x0.get_q()))

        # for k in range(1, self.N):
        #     self.acados_ocp_solver.constraints_set(k, 'lbx', lbx)
        #     self.acados_ocp_solver.constraints_set(k, 'ubx', ubx)

    # Systems dynamics:
    def __f(self, x, u):
        w_r = u[self.hparams.wr_idx]
        w_l = u[self.hparams.wl_idx]
        wheel_radius = self.hparams.wheel_radius
        wheel_separation = self.hparams.wheel_separation
        v = (wheel_radius/2)*(w_r+w_l)
        omega = (wheel_radius/wheel_separation)*(w_r-w_l)

        xdot = casadi.SX.zeros(self.nq)
        xdot[self.hparams.x_idx] = self.__x_dot(x,v,omega)
        xdot[self.hparams.y_idx] = self.__y_dot(x,v,omega)
        xdot[self.hparams.theta_idx] = self.__theta_dot(omega)
        return xdot

    def __x_dot(self, q, v, omega):
        b = self.hparams.b
        return v*casadi.cos(q[self.hparams.theta_idx])-omega*b*casadi.sin(q[self.hparams.theta_idx])

    def __y_dot(self, q, v, omega):
        b = self.hparams.b
        return v*casadi.sin(q[self.hparams.theta_idx])+omega*b*casadi.cos(q[self.hparams.theta_idx])
    
    def __theta_dot(self, omega):
        return omega
    
    def __h_i(self, q):
        n_obs = self.hparams.n_obstacles
        if n_obs > 0:
            p = casadi.SX.zeros((n_obs, 2))
            h_i = casadi.SX.zeros(4 + n_obs)
        else:
            h_i = casadi.SX.zeros(4)

        # Consider the robot distance from the bounds [ubx, lbx, uby, lby]
        distance_bounds = casadi.SX.zeros((4,1))
        distance_bounds[0] = self.hparams.x_upper_bound - q[self.hparams.x_idx]
        distance_bounds[1] = q[self.hparams.x_idx] - self.hparams.x_lower_bound
        distance_bounds[2] = self.hparams.y_upper_bound - q[self.hparams.y_idx] 
        distance_bounds[3] = q[self.hparams.y_idx] - self.hparams.y_lower_bound
        for i in range(4):
            h_i[i] = distance_bounds[i]

        # Consider the robot distance from obstacles, if obstacles are present
        if n_obs > 0:
            distance_vectors = casadi.SX.zeros((n_obs, 2))
            cbf_radius = self.hparams.rho_cbf + self.hparams.ds_cbf
            for i in range(n_obs):
                p[i, :] = self.hparams.obstacles_position[i, :]
                distance_vectors[i, self.hparams.x_idx] = q[self.hparams.x_idx] - p[i, self.hparams.x_idx]
                distance_vectors[i, self.hparams.y_idx] = q[self.hparams.y_idx] - p[i, self.hparams.y_idx]
                h_i[i + 4] = distance_vectors[i, self.hparams.x_idx]**2 + \
                        distance_vectors[i, self.hparams.y_idx]**2 - \
                        cbf_radius**2
        return h_i

    def __h_i_dot(self, q, u):
        x = q[self.hparams.x_idx]
        y = q[self.hparams.y_idx]
        v = self.hparams.wheel_radius * 0.5 * (u[self.hparams.wr_idx] + u[self.hparams.wl_idx])
        omega = (self.hparams.wheel_radius/self.hparams.wheel_separation)*(u[self.hparams.wr_idx]- u[self.hparams.wl_idx])
        return casadi.jacobian(self.__h_i(q), x) * self.__x_dot(q, v, omega) + \
               casadi.jacobian(self.__h_i(q), y) * self.__y_dot(q, v, omega)        

    
    def __create_acados_model(self) -> AcadosModel:
        # Setup CasADi expressions:
        q = casadi.SX.sym('q', self.nq)
        qdot = casadi.SX.sym('qdot', self.nq)
        u = casadi.SX.sym('u', self.nu)
        f_expl = self.__f(q, u)
        f_impl = qdot - f_expl

        # Create acados model:
        acados_model = AcadosModel()
        acados_model.name = 'tiago_kinematic_model'

        # System dynamics:
        acados_model.f_impl_expr = f_impl
        acados_model.f_expl_expr = f_expl

        # CBF constraints:
        con_h_expr = self.__h_i_dot(q, u) + self.hparams.gamma_cbf * self.__h_i(q)
        acados_model.con_h_expr = con_h_expr

        # Variables and params:
        acados_model.x = q
        acados_model.xdot = qdot
        acados_model.u = u

        return acados_model
    
    def __create_acados_cost(self) -> AcadosOcpCost:
        acados_cost = AcadosOcpCost()

        # Set wheighting matrices
        Q_mat = 2 * np.diag([self.hparams.q, self.hparams.q, 0.0]) # [x, y, theta]
        R_mat = 2 * 5 * np.diag([self.hparams.r, self.hparams.r]) # [wr, wl]

        acados_cost.cost_type   = 'LINEAR_LS'
        acados_cost.cost_type_e = 'LINEAR_LS'
        
        ny = self.nq + self.nu
        ny_e = self.nq

        acados_cost.W_e = self.hparams.q_factor * Q_mat
        acados_cost.W = scipy.linalg.block_diag(Q_mat,R_mat)

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

        # Linear inequality constraints on the control inputs:
        acados_constraints.idxbu = np.array([self.hparams.wr_idx, self.hparams.wl_idx])
        acados_constraints.lbu = np.array([self.hparams.w_max_neg, self.hparams.w_max_neg])
        acados_constraints.ubu = np.array([self.hparams.w_max, self.hparams.w_max])
        

        # Linear inequality constraints on the state:
        # acados_constraints.idxbx = np.array([self.hparams.x_idx, self.hparams.y_idx])
        # acados_constraints.lbx = np.zeros(len(acados_constraints.idxbx))
        # acados_constraints.ubx = np.zeros(len(acados_constraints.idxbx))
        acados_constraints.x0 = np.zeros(self.nq)

        # Linear constraints on driving and steering velocity expressed in term of wheel angular velocities
        D_mat = np.zeros((self.nu, self.nu))
        D_mat[0 , :self.nu] = self.hparams.wheel_radius * 0.5 * np.ones((1,2))
        D_mat[1, :self.nu] = (self.hparams.wheel_radius/self.hparams.wheel_separation) * np.array([1, -1])
        acados_constraints.D = D_mat
        acados_constraints.C = np.zeros((self.nu, self.nq))
        acados_constraints.lg = np.array([self.hparams.driving_vel_min, self.hparams.steering_vel_max_neg])
        acados_constraints.ug = np.array([self.hparams.driving_vel_max, self.hparams.steering_vel_max])

        # Nonlinear constraints (CBFs) (for both obstacles and configuration bounds):
        if self.hparams.n_obstacles > 0:
            acados_constraints.lh = np.zeros(self.hparams.n_obstacles + 4)
            acados_constraints.uh = np.zeros(self.hparams.n_obstacles + 4)
        else:
            acados_constraints.lh = np.zeros(4)
            acados_constraints.uh = np.zeros(4)

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
            configuration: Configuration,
            q_ref: np.array,
            u_ref: np.array):
        # Set parameters
        for k in range(self.N):
            self.acados_ocp_solver.set(k, 'y_ref', np.concatenate((q_ref[:, k], u_ref[:, k])))
        self.acados_ocp_solver.set(self.N, 'y_ref', q_ref[:, self.N])

        # Solve NLP
        self.u0 = self.acados_ocp_solver.solve_for_x0(configuration.get_q())
        self.tmp_time = self.acados_ocp_solver.get_stats('time_tot')
        if self.tmp_time > self.max_time:
            self.max_time = self.tmp_time
            self.idx_time = rospy.get_time()

    def get_command(self):
        return self.u0
