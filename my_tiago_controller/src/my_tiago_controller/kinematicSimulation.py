import numpy as np
import math
import rospy

from my_tiago_controller.kinematicModel import *
from my_tiago_controller.controllerManager import *
import my_tiago_controller.utils
class KinematicSimulation:
    def __init__(
            self,
            controller_manager: ControllerManager,
            dt: float,
            publish_data=True,):
        self.__iter = 0
        self.dt = dt
        self.kinematic_model = KinematicModel()
        self.controller_manager = controller_manager
        self.publish_data = publish_data

    def get_simulation_time(self):
        return self.dt * self.__iter
    
    def update(self):
        t = self.get_simulation_time()

        self.controller_manager.update()

        # Integrate using RK4:
        kinematic_model = self.kinematic_model
        self.controller_manager.configuration = my_tiago_controller.utils.integrate(
            kinematic_model,
            self.controller_manager.configuration,
            self.controller_manager.control_input,
            self.dt
        )
        # Publish data to ROS topics if specified:
        if self.publish_data:
            self.controller_manager.publish_command()
            self.controller_manager.publish_odometry()

        self.__iter = self.__iter + 1

        return True

def main():
    rospy.init_node('tiago_nmpc_controller', log_level=rospy.INFO)
    rospy.loginfo('Tiago control module [OK]')

    # Build controller manager
    controller_frequency = 40.0 # [Hz]
    dt = 1.0 / controller_frequency
    N_horizon = 5
    T_horizon = dt * 10.0 * N_horizon # [s]
    controller_manager = ControllerManager(
        controller_frequency=controller_frequency,
        nmpc_N=N_horizon,
        nmpc_T=T_horizon
    )

    # Setup kinematic simulation
    starting_configuration = np.array([0.0, 0.0, math.pi/2])
    controller_manager.init(starting_configuration)
    tiago_kinematic_simulation = KinematicSimulation(controller_manager, dt)
    rate = rospy.Rate(controller_frequency)
    while not rospy.is_shutdown():
        tiago_kinematic_simulation.update()

        rate.sleep()

    # Nsim = 100
    # nx = controller.acados_ocp.model.x.size()[0]
    # nu = controller.acados_ocp.model.u.size()[0]
    # yref = np.zeros((nx + nu,))
    # yref_N = np.zeros((nx,))



    # xcurrent = x0
    # simX[0, :] = xcurrent

    # # Init solver
    # for stage in range(N_horizon + 1):
    #     acados_ocp_solver.set(stage, "x", 0.0 * np.ones(xcurrent.shape))
    # for stage in range(N_horizon):
    #     acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # # Closed loop
    # for i in range(Nsim):

    #     # Set initial state constraint
    #     acados_ocp_solver.set(0, "lbx", xcurrent)
    #     acados_ocp_solver.set(0, "ubx", xcurrent)

    #     # Update yref
    #     for j in range(N_horizon):
    #         yref[:nx-1] = params.target_point
    #         acados_ocp_solver.set(j, "yref", yref)
    #     yref_N[:nx-1] = params.target_point
    #     acados_ocp_solver.set(N_horizon, "yref", yref_N)

    #     # Solve ocp
    #     status = acados_ocp_solver.solve()
    #     if status not in [0, 2]:
    #         acados_ocp_solver.print_statistics()
    #         plot_robot(
    #             np.linspace(0, T_horizon / N_horizon * i, i + 1),
    #             params.w_max,
    #             simU[:i, :],
    #             simX[: i + 1, :],
    #         )
    #         raise Exception(
    #             f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
    #         )

    #     if status == 2:
    #         print(
    #             f"acados acados_ocp_solver returned status {status} in closed loop instance {i} with {xcurrent}"
    #         )
    #     simU[i, :] = acados_ocp_solver.get(0, "u")

    #     # simulate system
    #     acados_integrator.set("x", xcurrent)
    #     acados_integrator.set("u", simU[i, :])

    #     status = acados_integrator.solve()
    #     if status != 0:
    #         raise Exception(
    #             f"acados integrator returned status {status} in closed loop instance {i}"
    #         )

    #     # update state
    #     xcurrent = acados_integrator.get("x")
    #     simX[i + 1, :] = xcurrent

    # # plot results
    # plot_robot(
    #     np.linspace(0, T_horizon / N_horizon * Nsim, Nsim + 1), [params.w_max, None], simU, simX
    # )