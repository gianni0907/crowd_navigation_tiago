import numpy as np
import scipy.linalg
import rospy
import nav_msgs.msg
import geometry_msgs.msg

from acados_template import AcadosModel, AcadosOcp, AcadosOcpConstraints, AcadosOcpCost, AcadosOcpOptions
from acados_template import AcadosOcpSolver, AcadosSimSolver

import casadi

from my_tiago_controller.Hparams import *
from my_tiago_controller.utils import *

class nmpc:
    def __init__(self, N, T, x0, params):
        # Size of state and input:
        self.nq = 3
        self.nu = 2 # right and left wheel angular velocities

        # Parameters:
        self.hparams = params

        # Number of control intervals:
        self.N = N

        # Horizon duration:
        self.T = T

        # Initial configuration
        self.x0 = x0

        # Create acados_ocp object to formulate the OCP
        self.acados_ocp = self.__create_acados_ocp(self.N,self.T)

        # Setup velocity command publisher:
        cmd_vel_topic = '/mobile_base_controller/cmd_vel'
        self.cmd_vel_publisher = rospy.Publisher(cmd_vel_topic, geometry_msgs.msg.Twist, queue_size=5)

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
        R_mat = 2 * 5 * np.diag([1e-2,1e-2]) # [wr, wl]

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
        Vu[self.nq : ny, 0 :self.nu] = np.eye(self.nu)
        acados_cost.Vu = Vu

        acados_cost.Vx_e = np.eye(ny_e)

        acados_cost.yref = np.zeros(ny)
        acados_cost.yref_e = np.zeros(ny_e)
        
        return acados_cost
    
    def __create_acados_constraints(self) -> AcadosOcpConstraints:

        acados_constraints = AcadosOcpConstraints()

        # Linear inequality constraints on the control inputs:
        acados_constraints.idxbu = np.array([self.hparams.wr_idx,self.hparams.wl_idx])
        acados_constraints.lbu = np.array([self.hparams.w_max_neg,self.hparams.w_max_neg])
        acados_constraints.ubu = np.array([self.hparams.w_max,self.hparams.w_max])

        # Linear inequality constraints on the state:
        acados_constraints.idxbx = np.array([self.hparams.x_idx,self.hparams.y_idx])
        acados_constraints.lbx = np.array([self.hparams.lbx,self.hparams.lbx])
        acados_constraints.ubx = np.array([self.hparams.ubx,self.hparams.ubx])
        acados_constraints.x0 = self.x0

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
        
    def publish_velocity_command(self, command):
        # Set wheel angular velocity commands
        w_r = command[self.hparams.wr_idx]
        w_l = command[self.hparams.wl_idx]
        wheel_radius = self.hparams.wheel_radius
        wheel_separation = self.hparams.wheel_separation

        # Compute driving and steering velocity commands
        v = (wheel_radius/2)*(w_r+w_l)
        omega = (wheel_radius/wheel_separation)*(w_r-w_l)

        # Create a twist ROS message:
        cmd_vel_msg = geometry_msgs.msg.Twist()
        cmd_vel_msg.linear.x = v
        cmd_vel_msg.linear.y = 0.0
        cmd_vel_msg.linear.z = 0.0
        cmd_vel_msg.angular.x = 0.0
        cmd_vel_msg.angular.y = 0.0
        cmd_vel_msg.angular.z = omega

        # Publish wheel velocity commands
        self.cmd_vel_publisher.publish(cmd_vel_msg)

def main():
    rospy.init_node('tiago_nmpc_controller', log_level=rospy.INFO)
    rospy.loginfo('Tiago control module [OK]')
    controller_frequency = 10.0 # [Hz]
    dt = 1.0 / controller_frequency
    rate = rospy.Rate(controller_frequency/10.0)

    N_horizon = 5
    T_horizon = dt * 1 * N_horizon # [s]
    x0 = np.array([0.0, 0.0, 0.0])
    params = Hparams()
    controller = nmpc(N_horizon, T_horizon, x0, params)

    acados_ocp_solver = AcadosOcpSolver(
        controller.acados_ocp, json_file="acados_ocp_" + controller.acados_ocp.model.name + ".json"
    )
    acados_integrator = AcadosSimSolver(
        controller.acados_ocp, json_file="acados_ocp_" + controller.acados_ocp.model.name + ".json"
    )

    Nsim = 100
    nx = controller.acados_ocp.model.x.size()[0]
    nu = controller.acados_ocp.model.u.size()[0]
    yref = np.zeros((nx + nu,))
    yref_N = np.zeros((nx,))

    simX = np.ndarray((Nsim + 1, nx))
    simU = np.ndarray((Nsim, nu))

    xcurrent = x0
    simX[0, :] = xcurrent
    print("Init configuration -----------")
    print(simX[0,:])

    # Init solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(xcurrent.shape))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # Closed loop
    for i in range(Nsim):

        # Set initial state constraint
        acados_ocp_solver.set(0, "lbx", xcurrent)
        acados_ocp_solver.set(0, "ubx", xcurrent)

        # Update yref
        for j in range(N_horizon):
            yref[:nx-1] = params.target_point
            acados_ocp_solver.set(j, "yref", yref)
        yref_N[:nx-1] = params.target_point
        acados_ocp_solver.set(N_horizon, "yref", yref_N)

        # Solve ocp
        status = acados_ocp_solver.solve()
        # if status not in [0, 2]:
        #     acados_ocp_solver.print_statistics()
        #     plot_robot(
        #         np.linspace(0, T_horizon / N_horizon * i, i + 1),
        #         params.w_max,
        #         simU[:i, :],
        #         simX[: i + 1, :],
        #     )
        #     raise Exception(
        #         f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
        #     )

        # if status == 2:
        #     print(
        #         f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
        #     )
        simU[i, :] = acados_ocp_solver.get(0, "u")
        print(i, "-th control input --------")
        print(simU[i,:])
        if i==0:
            controller.publish_velocity_command(simU[i,:])
        else:
            command = np.array([simU[0,0],-simU[0,0]])
            controller.publish_velocity_command(command)

        # simulate system
        acados_integrator.set("x", xcurrent)
        acados_integrator.set("u", simU[i, :])

        status = acados_integrator.solve()
        # if status != 0:
        #     raise Exception(
        #         f"acados integrator returned status {status} in closed loop instance {i}"
        #     )

        # update state
        xcurrent = acados_integrator.get("x")
        simX[i + 1, :] = xcurrent
        print(i+1, "-th configuration ---------")
        print(xcurrent)
        rate.sleep()
    # plot results
    plot_robot(
        np.linspace(0, T_horizon / N_horizon * Nsim, Nsim + 1), [params.w_max, None], simU, simX
    )
