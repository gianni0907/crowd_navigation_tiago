import numpy as np
import rospy

from my_tiago_controller.kinematicModel import *
from my_tiago_controller.controllerManager import *
import my_tiago_controller.utils

class KinematicSimulation:
    def __init__(
            self,
            controller_manager: ControllerManager,
            dt: float,
            publish_data=True):
        self.dt = dt
        self.kinematic_model = KinematicModel()
        self.controller_manager = controller_manager
        self.publish_data = publish_data
    
    def update(self):
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

        return True
    
def main():
    # Build controller manager
    controller_frequency = 15.0 # [Hz]
    dt = 1.0 / controller_frequency
    N_horizon = 5
    T_horizon = dt * 1.0 * N_horizon # [s]
    controller_manager = ControllerManager(
        controller_frequency=controller_frequency,
        nmpc_N=N_horizon,
        nmpc_T=T_horizon
    )

    # Setup kinematic simulation
    starting_configuration = np.array([0.0, 0.0, 0.0])
    controller_manager.init(starting_configuration)
    tiago_kinematic_simulation = KinematicSimulation(controller_manager, dt)

    # Set variables for plots
    N_sim = 100
    iter = 0
    x_real = np.ndarray((N_sim + 1, controller_manager.nmpc_controller.nq))
    x_sim = np.ndarray((N_sim + 1, controller_manager.nmpc_controller.nq))
    u_sim = np.ndarray((N_sim, controller_manager.nmpc_controller.nu))
    x_sim[0, :] = starting_configuration
    x_real[0, :] = starting_configuration
    save = True
    save_dir = '/home/gianba/thesis_ws/src/crowd_navigation_tiago/plots'

    rospy.init_node('tiago_nmpc_controller', log_level=rospy.INFO)
    rospy.loginfo('Tiago control module [OK]')
    rate = rospy.Rate(controller_frequency)

    print("Init configuration ------------")
    print(starting_configuration)

    while not(rospy.is_shutdown()) and (iter < N_sim):
        tiago_kinematic_simulation.update()
        
        x_real[iter + 1, :] = tiago_kinematic_simulation.controller_manager.get_latest_configuration()

        u_sim[iter, :] = tiago_kinematic_simulation.controller_manager.control_input
        print(iter,"-th command ***********")
        print(u_sim[iter, :])
        
        x_sim[iter + 1, :] = tiago_kinematic_simulation.controller_manager.configuration
        print(iter+1,"-th configuration -----")
        print(x_sim[iter + 1, :])

        iter = iter +1
        rate.sleep()

    plot_robot(
        np.linspace(0, T_horizon / N_horizon * N_sim, N_sim + 1),
        [controller_manager.hparams.w_max, controller_manager.hparams.w_max],
        u_sim,
        x_sim,
        x_real,
        save,
        save_dir
    )