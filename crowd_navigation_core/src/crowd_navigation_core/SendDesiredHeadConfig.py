import rospy

import crowd_navigation_msgs.srv

def send_desired_head_config(pan, tilt):
    rospy.wait_for_service("SetDesiredHeadConfig")

    try:
        set_desired_head_config_service = rospy.ServiceProxy(
            'SetDesiredHeadConfig',
            crowd_navigation_msgs.srv.SetDesiredHeadConfig
        )

        response = set_desired_head_config_service(pan, tilt)
        
        if response.success:
            rospy.loginfo("Desired head configuration successfully set")
        else:
            rospy.loginfo("Could not set desired head configuration")

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def main():
    rospy.init_node('tiago_send_desired_head_config', log_level=rospy.INFO)
    rospy.loginfo('TIAGo send desired head configuration module [OK]')

    pan = rospy.get_param('/pan') # [rad]
    tilt = rospy.get_param('/tilt') # [rad]
    # Service request:
    rospy.loginfo("Sending desired head configuration")
    send_desired_head_config(pan, tilt)
    

