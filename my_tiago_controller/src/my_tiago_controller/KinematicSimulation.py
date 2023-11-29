import numpy as np
import os
import rospy

from my_tiago_controller.KinematicModel import *
from my_tiago_controller.ControllerManager import *
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

        # Setup odometry publisher:
        odometry_topic = "/odom"
        self.odometry_publisher = rospy.Publisher(odometry_topic, nav_msgs.msg.Odometry, queue_size=1)

    def publish_odometry(self):
        # Publish odometry:
        odom_message = nav_msgs.msg.Odometry()
        odom_message.header.stamp = rospy.Time.now()
        odom_message.header.frame_id = 'odom'
        odom_message.child_frame_id = 'base_link'

        configuration = self.controller_manager.configuration
        control_input = self.controller_manager.control_input
        wheel_radius = self.controller_manager.hparams.wheel_radius
        wheel_separation = self.controller_manager.hparams.wheel_separation
        x_idx = self.controller_manager.hparams.x_idx
        y_idx = self.controller_manager.hparams.y_idx
        theta_idx = self.controller_manager.hparams.theta_idx
        wr_idx = self.controller_manager.hparams.wr_idx
        wl_idx = self.controller_manager.hparams.wl_idx

        odom_message.pose.pose.position.x = configuration[x_idx]
        odom_message.pose.pose.position.y = configuration[y_idx]
        odom_message.pose.pose.position.z = 0.0
        orientation = scipy.spatial.transform.Rotation.from_dcm(
            [[math.cos(configuration[theta_idx]), -math.sin(configuration[theta_idx]), 0.0],
             [math.sin(configuration[theta_idx]),  math.cos(configuration[theta_idx]), 0.0],
             [                               0.0,                                 0.0, 1.0]]
        ).as_quat()
        odom_message.pose.pose.orientation.x = orientation[0]
        odom_message.pose.pose.orientation.y = orientation[1]
        odom_message.pose.pose.orientation.z = orientation[2]
        odom_message.pose.pose.orientation.w = orientation[3]

        odom_message.twist.twist.linear.x = (wheel_radius/2)*(control_input[wr_idx]+control_input[wl_idx])
        odom_message.twist.twist.linear.y = 0.0
        odom_message.twist.twist.linear.z = 0.0
        odom_message.twist.twist.angular.x = 0.0
        odom_message.twist.twist.angular.y = 0.0
        odom_message.twist.twist.angular.z = (wheel_radius/wheel_separation)*(control_input[wr_idx]-control_input[wl_idx])

        self.odometry_publisher.publish(odom_message)
    
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
            self.publish_odometry()

        return True
    
def main():
    # Build controller manager
    controller_frequency = 50.0 # [Hz]
    dt = 1.0 / controller_frequency
    N_horizon = 25
    T_horizon = dt * N_horizon # [s]
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
    N_sim = 250
    iter = 0
    x_real = np.ndarray((N_sim + 1, controller_manager.nmpc_controller.nq))
    x_sim = np.ndarray((N_sim + 1, controller_manager.nmpc_controller.nq))
    u_sim = np.ndarray((N_sim, controller_manager.nmpc_controller.nu))
    x_sim[0, :] = starting_configuration
    x_real[0, :] = starting_configuration
    save = True
    save_dir = '/tmp/crowd_navigation_tiago'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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