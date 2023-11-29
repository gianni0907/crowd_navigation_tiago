import os
import rospy
import rosbag
import nav_msgs.msg
import geometry_msgs.msg

class Logger:
    def __init__(self, topic=None, path=None):
        if path==None:
            raise Exception(
                f"Path to save rosbag not specified"
            )
        if topic==None:
            raise Exception(
                f"Topic to record rosbag not specified"
            )
        
        self.topic = topic
        self.bag = rosbag.Bag(path, 'w')
        
    def logger_callback(self, msg):
        rospy.loginfo("Received message: %s", msg)
        self.bag.write(self.topic, msg)

    def start_logging(self):
        if self.topic=="/mobile_base_controller/cmd_vel":
            rospy.Subscriber(self.topic, geometry_msgs.msg.Twist, self.logger_callback)
        elif self.topic=="/mobile_base_controller/odom" or self.topic=="/odom":
            rospy.Subscriber(self.topic, nav_msgs.msg.Odometry, self.logger_callback)
        rospy.spin()


    def stop_logging(self):
        self.bag.close()

def main():
    rospy.init_node('logger', anonymous=True)

    # Access the parameters
    topic = rospy.get_param('/topic')
    filename = rospy.get_param('/bagname')

    save_dir = '/tmp/crowd_navigation_tiago/bagfiles'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    bag_path = os.path.join(save_dir, filename)
    
    logger = Logger(topic,bag_path)

    try:
        logger.start_logging()
    except rospy.ROSInterruptException as e:
        rospy.logwarn("Logger initialization failed")
        rospy.logwarn('{}'.format(e))
    finally:
        logger.stop_logging()