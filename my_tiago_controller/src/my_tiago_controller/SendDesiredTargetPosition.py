
import numpy as np
import rospy

import my_tiago_msgs.srv

def send_desired_target_position(desired_position):
    rospy.wait_for_service('SetDesiredTargetPosition')

    try:
        set_desired_target_position_service = rospy.ServiceProxy(
            'SetDesiredTargetPosition',
            my_tiago_msgs.srv.SetDesiredTargetPosition
        )

        x = desired_position[0]
        y = desired_position[1]

        response = set_desired_target_position_service(x,y)

        if response.success:
            rospy.loginfo('Desired target position successfully set')
        else:
            rospy.loginfo('Could not set desired target position')

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def main():
    rospy.init_node('tiago_send_desired_target_position', log_level=rospy.INFO)
    rospy.loginfo('TIAGo send desired target position module [OK]')

    x_des = rospy.get_param('/x_des') # [m]
    y_des = rospy.get_param('/y_des') # [m]
    target_position = np.array([x_des, y_des])
    # Service request:
    rospy.loginfo("Sending desired target position")
    send_desired_target_position(target_position)