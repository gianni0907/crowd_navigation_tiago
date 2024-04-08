#!/usr/bin/env python

import rospy

import math

import tf_conversions
import tf2_ros

import gazebo_msgs.msg
import geometry_msgs.msg

def wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))

class Configuration:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
    
    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.theta)

    @staticmethod
    def set_from_tf_transform(transform):
        
        x = transform.transform.translation.x
        y = transform.transform.translation.y
        q = transform.transform.rotation
        theta = math.atan2(
          2.0 * (q.w * q.z + q.x * q.y),
          1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        config = Configuration(x, y, theta)
        return config

class RobotLocalization():
    def __init__(self, robot_name):

        self.robot_name = robot_name

        self.configuration = None

        # Setting up reference frames:
        self.odom_frame = 'odom'
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'

        # Setting up TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        states_topic = "/gazebo/model_states"

        # Setting up the listener to the robot configuration:
        rospy.Subscriber(
            states_topic,
            gazebo_msgs.msg.ModelStates,
            self.gazebo_model_states_callback
        )

    def gazebo_model_states_callback(self, gazebo_model_states_msg):
        self.robot_name = "tiago"
        if self.robot_name in gazebo_model_states_msg.name:
            tiago_idx = gazebo_model_states_msg.name.index(self.robot_name)
            p = gazebo_model_states_msg.pose[tiago_idx].position
            q = gazebo_model_states_msg.pose[tiago_idx].orientation
            self.configuration =  Configuration(
                p.x,
                p.y,
                math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                )
            )

    def broadcast_transform(self, configuration_odom, configuration_world_gazebo):
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame

        angle_diff = configuration_world_gazebo.theta - configuration_odom.theta

        t.transform.translation.x = \
            configuration_world_gazebo.x \
            - (math.cos(angle_diff)*configuration_odom.x \
               - math.sin(angle_diff)*configuration_odom.y)
        t.transform.translation.y = \
            configuration_world_gazebo.y \
                - (math.sin(angle_diff)*configuration_odom.x \
                   + math.cos(angle_diff)*configuration_odom.y)
        t.transform.translation.z = 0.0
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, wrap_angle(angle_diff))
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    
    def start(self):
        rate = rospy.Rate(100) # 100 Hz

        while not rospy.is_shutdown():
            # Read robot configuration from odom:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.odom_frame, self.base_footprint_frame, rospy.Time()
                )
                configuration_odom = Configuration.set_from_tf_transform(transform)

                self.broadcast_transform(configuration_odom, self.configuration)
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                rate.sleep()
                continue

            # Keep controller frequency at specified rate:
            rate.sleep()


def main():
    rospy.init_node('robot_localization', log_level=rospy.INFO)

    robot_localization = RobotLocalization("tiago")
    robot_localization.start()


if __name__ == '__main__':
    main()