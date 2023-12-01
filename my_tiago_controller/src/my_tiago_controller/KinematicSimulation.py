import numpy as np
import os
import rospy

from my_tiago_controller.KinematicModel import *
from my_tiago_controller.ControllerManager import *
from my_tiago_controller.Logger import *
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
        ### NOTE: The publication of the velocity commands and odometry
        ###       is not required for the kinematic simulation (no interaction with Gazebo)
        ###       Anyway, such publishers are used to publish commands and odometry,
        ###       on topics independent from Gazebo, in order to record them in rosbags and
        ###       make a comparison between the pure kinematic simulation and the dynamic one (on Gazebo)
        # Setup publisher for wheel velocity commands:
        cmd_vel_topic = '/cmd_vel'
        self.cmd_vel_publisher = rospy.Publisher(cmd_vel_topic, geometry_msgs.msg.Twist, queue_size=1)

        # Setup odometry publisher:
        odom_topic = "/odom"
        self.odometry_publisher = rospy.Publisher(odom_topic, nav_msgs.msg.Odometry, queue_size=1)

    def publish_command(self):
        # Set wheel angular velocity commands
        w_r = self.controller_manager.control_input[self.controller_manager.hparams.wr_idx]
        w_l = self.controller_manager.control_input[self.controller_manager.hparams.wl_idx]
        wheel_radius = self.controller_manager.hparams.wheel_radius
        wheel_separation = self.controller_manager.hparams.wheel_separation

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
            self.publish_command()
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

    rospy.init_node('tiago_nmpc_controller', log_level=rospy.INFO)
    rospy.loginfo('TIAGo control module [OK]')
    rate = rospy.Rate(controller_frequency)

    # Setup kinematic simulation
    starting_configuration = np.array([0.0, 0.0, 0.0])
    controller_manager.init(starting_configuration)
    tiago_kinematic_simulation = KinematicSimulation(controller_manager, dt)

    # Set variables for plots
    N_sim = 400
    iter = 0
    x_sim = np.ndarray((N_sim + 1, controller_manager.nmpc_controller.nq))
    u_sim = np.ndarray((N_sim, controller_manager.nmpc_controller.nu))
    x_sim[0, :] = starting_configuration
    save = True
    save_dir = '/tmp/crowd_navigation_tiago/plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Setup loggers for bagfiles
    log = False
    bag_dir = '/tmp/crowd_navigation_tiago/bagfiles'
    if not os.path.exists(bag_dir):
        os.makedirs(bag_dir)

    if log:
        odom_topic = "/odom"
        cmd_vel_topic = "/cmd_vel"
        odom_bagname = "odometry_simulation.bag"
        cmd_vel_bagname = "commands_simulation.bag"
    
        odom_bag = os.path.join(bag_dir, odom_bagname)
        cmd_vel_bag = os.path.join(bag_dir, cmd_vel_bagname)

        odom_logger = Logger(odom_topic, odom_bag)
        cmd_vel_logger = Logger(cmd_vel_topic, cmd_vel_bag)

        # Start loggers
        odom_logger.start_logging()
        cmd_vel_logger.start_logging()
    
    print("Init configuration ------------")
    print(starting_configuration)

    try:
        while not(rospy.is_shutdown()) and (iter < N_sim):
            tiago_kinematic_simulation.update()
        
            u_sim[iter, :] = tiago_kinematic_simulation.controller_manager.control_input        
            x_sim[iter + 1, :] = tiago_kinematic_simulation.controller_manager.configuration

            iter = iter +1
            rate.sleep()
    except rospy.ROSInterruptException as e:
        rospy.logwarn("ROS node shutting down")
        rospy.logwarn('{}'.format(e))
    finally:
        if log:
            # Stop loggers
            odom_logger.stop_logging()
            cmd_vel_logger.stop_logging()

    plot_robot(
        np.linspace(0, T_horizon / N_horizon * N_sim, N_sim + 1),
        [controller_manager.hparams.w_max, controller_manager.hparams.w_max],
        u_sim,
        x_sim,
        save,
        save_dir
    )