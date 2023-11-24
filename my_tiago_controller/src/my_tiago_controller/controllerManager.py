import math
import matplotlib.pyplot as plt
import scipy.spatial.transform

import rospy
import nav_msgs.msg
import std_msgs.msg

from my_tiago_controller.hparams import *
from my_tiago_controller.kinematicModel import *
from my_tiago_controller.nmpc import *

class ControllerManager:
    def __init__(
            self,
            controller_frequency,
            nmpc_N,
            nmpc_T):
        self.controller_frequency = controller_frequency

        # Wheel velocity command topics:
        w_r_command_topic = "/my_tiago_controller/right_wheel_controller/command"
        w_l_command_topic = "/my_tiago_controller/left_wheel_controller/command"

        # Setup publishers for wheel velocity commands:
        self.w_r_command_publisher = rospy.Publisher(w_r_command_topic, std_msgs.msg.Float64, queue_size=1)
        self.w_l_command_publisher = rospy.Publisher(w_l_command_topic, std_msgs.msg.Float64, queue_size=1)

        # Setup odometry publisher:
        odometry_topic = 'odom'
        self.odometry_publisher = rospy.Publisher(odometry_topic,nav_msgs.msg.Odometry, queue_size=1)

        # NMPC:
        self.dt = 1.0 / self.controller_frequency
        self.hparams = Hparams()
        self.target_point = self.hparams.target_point
        self.nmpc_controller = NMPC(nmpc_N, nmpc_T)

    def init(self, configuration):
        # Init robot configuration
        self.configuration = configuration
        print(self.configuration)

    def update(self):
        self.nmpc_controller.init(self.configuration)
        q_ref = np.zeros((self.nmpc_controller.nq, self.nmpc_controller.N+1))
        for k in range(self.nmpc_controller.N):
            q_ref[:self.nmpc_controller.nq - 1, k] = self.target_point
        u_ref = np.zeros((self.nmpc_controller.nu, self.nmpc_controller.N))
        q_ref[:self.nmpc_controller.nq - 1, self.nmpc_controller.N] = self.target_point

        try:
            self.nmpc_controller.update(
                self.configuration,
                q_ref,
                u_ref
            )
            self.control_input = self.nmpc_controller.get_command()
            print(q_ref)
            print(u_ref)
            print(self.configuration)
            print(self.control_input)
        except Exception as e:
            rospy.logwarn("NMPC solver failed")
            rospy.logwarn('{}'.format(e))
            self.control_input = np.array([0.0, 0.0])
        
    def publish_command(self):
        # Set wheel angular velocity commands
        w_r_command = self.control_input[self.hparams.wr_idx]
        w_l_command = self.control_input[self.hparams.wl_idx]

        # Publish wheel velocity commands
        self.w_r_command_publisher.publish(w_r_command)
        self.w_l_command_publisher.publish(w_l_command)

    def publish_odometry(self):
        # Publish odometry
        odom_message = nav_msgs.msg.Odometry()
        odom_message.header.stamp = rospy.Time.now()
        odom_message.header.frame_id = 'odom'
        odom_message.child_frame_id = 'base_link'

        odom_message.pose.pose.position.x = self.configuration[self.hparams.x_idx]
        odom_message.pose.pose.position.y = self.configuration[self.hparams.y_idx]
        odom_message.pose.pose.position.z = 0.0
        # orientation = scipy.spatial.transform.Rotation.from_matrix(
        #     [[math.cos(self.configuration[self.hparams.theta_idx]), -math.sin(self.configuration[self.hparams.theta_idx]), 0.0],
        #      [math.sin(self.configuration[self.hparams.theta_idx]),  math.cos(self.configuration[self.hparams.theta_idx]), 0.0],
        #      [                                                 0.0,                                                   0.0, 1.0]]
        # ).as_quat()
        # odom_message.pose.pose.orientation.x = orientation[0]
        # odom_message.pose.pose.orientation.y = orientation[1]
        # odom_message.pose.pose.orientation.z = orientation[2]
        # odom_message.pose.pose.orientation.w = orientation[3]
        
        self.odometry_publisher.publish(odom_message)