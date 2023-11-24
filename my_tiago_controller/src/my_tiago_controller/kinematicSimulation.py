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
            publish_data=True):
        self.__iter = 0
        self.dt = dt
        self.kinematic_model = KinematicModel()
        self.controller_manager = controller_manager
        self.publish_data = publish_data

    def get_simulation_time(self):
        return self.dt * self.__iter
    
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
    starting_configuration = np.array([0.0, 0.0, 0.0])
    controller_manager.init(starting_configuration)
    tiago_kinematic_simulation = KinematicSimulation(controller_manager, dt)
    rate = rospy.Rate(controller_frequency)
    while not rospy.is_shutdown():
        tiago_kinematic_simulation.update()

        rate.sleep()