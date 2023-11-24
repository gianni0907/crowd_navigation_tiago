import numpy as np
import scipy.linalg

from acados_template import AcadosModel, AcadosOcp, AcadosOcpConstraints, AcadosOcpCost, AcadosOcpOptions, AcadosOcpSolver

import casadi

from my_tiago_controller.hparams import *
from my_tiago_controller.utils import *

class NMPC:
    def __init__(self, N, T):
        # Size of state and input:
        self.nq = 3
        self.nu = 2 # right and left wheel angular velocities

        # Parameters:
        self.hparams = Hparams()

        # Number of control intervals:
        self.N = N

        # Horizon duration:
        self.T = T

        # Setup solver:
        self.acados_ocp_solver = self.__create_acados_ocp_solver(self.N,self.T)
        
    def init(self, x0: np.array):
        lbx = np.array([self.hparams.lower_bound, self.hparams.lower_bound])
        ubx = np.array([self.hparams.upper_bound, self.hparams.upper_bound])

        for k in range(self.N):
            self.acados_ocp_solver.set(k, 'x', np.array(x0))
            self.acados_ocp_solver.set(k, 'u', np.zeros(self.nu))
        self.acados_ocp_solver.set(self.N, 'x', np.array(x0))

        for k in range(1, self.N):
            self.acados_ocp_solver.constraints_set(k, 'lbx', lbx)
            self.acados_ocp_solver.constraints_set(k, 'ubx', ubx)

    def __wrap_angle(self, theta):
        return casadi.atan2(casadi.sin(theta), casadi.cos(theta))

    # Systems dynamics:
    def __f(self, x, u):
        xdot = casadi.SX.zeros(self.nq)
        xdot[self.hparams.x_idx] = self.__x_dot(x,u)
        xdot[self.hparams.y_idx] = self.__y_dot(x,u)
        xdot[self.hparams.theta_idx] = self.__theta_dot(u)
        return xdot

    def __x_dot(self, q, u):
        return casadi.cos(q[self.hparams.theta_idx])*(self.hparams.wheel_radius/2)*(u[self.hparams.wr_idx]+u[self.hparams.wl_idx])

    def __y_dot(self, q, u):
        return casadi.sin(q[self.hparams.theta_idx])*(self.hparams.wheel_radius/2)*(u[self.hparams.wr_idx]+u[self.hparams.wl_idx])
    
    def __theta_dot(self, u):
        return (self.hparams.wheel_radius/self.hparams.wheel_separation)*(u[self.hparams.wr_idx]-u[self.hparams.wl_idx])
    
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

        # Variables and params:
        acados_model.x = q
        acados_model.xdot = qdot
        acados_model.u = u

        return acados_model
    
    def __create_acados_cost(self) -> AcadosOcpCost:
        acados_cost = AcadosOcpCost()

        # Set wheighting matrices
        Q_mat = 2 * np.diag([1e3, 1e3, 0.0]) # [x, y, theta]
        R_mat = 2 * 5 * np.diag([1e-2, 1e-2]) # [wr, wl]

        acados_cost.cost_type   = 'LINEAR_LS'
        acados_cost.cost_type_e = 'LINEAR_LS'
        
        ny = self.nq + self.nu
        ny_e = self.nq

        acados_cost.W_e = Q_mat
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
        acados_constraints.idxbu = np.array([self.hparams.wr_idx,self.hparams.wl_idx])
        acados_constraints.lbu = np.array([self.hparams.w_max_neg,self.hparams.w_max_neg])
        acados_constraints.ubu = np.array([self.hparams.w_max,self.hparams.w_max])

        # Linear inequality constraints on the state:
        acados_constraints.idxbx = np.array([self.hparams.x_idx,self.hparams.y_idx])
        acados_constraints.lbx = np.zeros(len(acados_constraints.idxbx))
        acados_constraints.ubx = np.zeros(len(acados_constraints.idxbx))
        acados_constraints.x0 = np.zeros(self.nq)

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
            configuration: np.array,
            q_ref: np.array,
            u_ref: np.array):
        # Set parameters
        for k in range(self.N):
            self.acados_ocp_solver.set(k, 'yref', np.concatenate((q_ref[:, k], u_ref[:, k])))
        self.acados_ocp_solver.set(self.N, 'yref',q_ref[:, self.N])

        # Solve NLP
        self.u0 = self.acados_ocp_solver.solve_for_x0(configuration)

    def get_command(self):
        return self.u0
