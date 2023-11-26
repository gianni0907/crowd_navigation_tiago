import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import rospy
import nav_msgs.msg
import geometry_msgs.msg

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
        cmd_vel_topic = '/mobile_base_controller/cmd_vel'

        # Setup publishers for wheel velocity commands:
        self.cmd_vel_publisher = rospy.Publisher(cmd_vel_topic, geometry_msgs.msg.Twist, queue_size=1)

        # Setup odometry publisher:
        odometry_topic = '/odom'
        self.odometry_publisher = rospy.Publisher(odometry_topic,nav_msgs.msg.Odometry, queue_size=1)

        # NMPC:
        self.dt = 1.0 / self.controller_frequency
        self.hparams = Hparams()
        self.target_point = self.hparams.target_point
        self.nmpc_controller = NMPC(nmpc_N, nmpc_T)

    def init(self, configuration):
        # Init robot configuration
        self.configuration = configuration
        self.nmpc_controller.init(self.configuration)

    def update(self):
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
        except Exception as e:
            rospy.logwarn("NMPC solver failed")
            rospy.logwarn('{}'.format(e))
            self.control_input = np.array([0.0, 0.0])
        
    def publish_command(self):
        # Set wheel angular velocity commands
        w_r = self.control_input[self.hparams.wr_idx]
        w_l = self.control_input[self.hparams.wl_idx]
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

    def publish_odometry(self):
        # Publish odometry
        odom_message = nav_msgs.msg.Odometry()
        odom_message.header.stamp = rospy.Time.now()
        odom_message.header.frame_id = 'odom'
        odom_message.child_frame_id = 'base_link'

        odom_message.pose.pose.position.x = self.configuration[self.hparams.x_idx]
        odom_message.pose.pose.position.y = self.configuration[self.hparams.y_idx]
        odom_message.pose.pose.position.z = 0.0
        orientation = R.from_dcm(
            [[math.cos(self.configuration[self.hparams.theta_idx]), -math.sin(self.configuration[self.hparams.theta_idx]), 0.0],
             [math.sin(self.configuration[self.hparams.theta_idx]),  math.cos(self.configuration[self.hparams.theta_idx]), 0.0],
             [                                                 0.0,                                                   0.0, 1.0]]
        ).as_quat()
        odom_message.pose.pose.orientation.x = orientation[0]
        odom_message.pose.pose.orientation.y = orientation[1]
        odom_message.pose.pose.orientation.z = orientation[2]
        odom_message.pose.pose.orientation.w = orientation[3]
        
        w_r = self.control_input[self.hparams.wr_idx]
        w_l = self.control_input[self.hparams.wl_idx]
        wheel_radius = self.hparams.wheel_radius
        wheel_separation = self.hparams.wheel_separation
        v = (wheel_radius/2)*(w_r+w_l)
        omega = (wheel_radius/wheel_separation)*(w_r-w_l)

        odom_message.twist.twist.linear.x =  v
        odom_message.twist.twist.linear.y = 0.0
        odom_message.twist.twist.linear.z = 0.0
        odom_message.twist.twist.angular.x = 0.0
        odom_message.twist.twist.angular.y = 0.0
        odom_message.twist.twist.angular.z = omega
        self.odometry_publisher.publish(odom_message)