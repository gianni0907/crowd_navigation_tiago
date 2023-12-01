import rospy
import nav_msgs.msg
import geometry_msgs.msg

from scipy.spatial.transform import Rotation as R

from my_tiago_controller.Hparams import *
from my_tiago_controller.KinematicModel import *
from my_tiago_controller.NMPC import *
from my_tiago_controller.Logger import *

import my_tiago_msgs.srv

class ControllerManager:
    def __init__(
            self,
            controller_frequency,
            nmpc_N,
            nmpc_T):
        self.controller_frequency = controller_frequency

        # Setup publisher for wheel velocity commands:
        cmd_vel_topic = '/mobile_base_controller/cmd_vel'
        self.cmd_vel_publisher = rospy.Publisher(
            cmd_vel_topic,
            geometry_msgs.msg.Twist,
            queue_size=1
        )

        # Setup odometry listener
        self.odom_topic = '/mobile_base_controller/odom'
        self.odometry_listener = rospy.Subscriber(
            self.odom_topic,
            nav_msgs.msg.Odometry,
            self.odom_callback
        )
        
        # NMPC:
        self.dt = 1.0 / self.controller_frequency
        self.hparams = Hparams()
        self.target_position = np.zeros((2,))
        self.nmpc_controller = NMPC(nmpc_N, nmpc_T)

        self.set_desired_target_position_srv = rospy.Service(
            'SetDesiredTargetPosition',
            my_tiago_msgs.srv.SetDesiredTargetPosition,
            self.set_desired_target_position_request
        )

    def init(self):
        # Init robot configuration
        self.nmpc_controller.init(self.configuration)

    def update(self):
        q_ref = np.zeros((self.nmpc_controller.nq, self.nmpc_controller.N+1))
        for k in range(self.nmpc_controller.N):
            q_ref[:self.nmpc_controller.nq - 1, k] = self.target_position
        u_ref = np.zeros((self.nmpc_controller.nu, self.nmpc_controller.N))
        q_ref[:self.nmpc_controller.nq - 1, self.nmpc_controller.N] = self.target_position

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

    def odom_callback(self, msg):
        self.configuration = np.zeros((self.nmpc_controller.nq))
        self.configuration[self.hparams.x_idx] = msg.pose.pose.position.x
        self.configuration[self.hparams.y_idx] = msg.pose.pose.position.y

        theta = R.from_quat([[msg.pose.pose.orientation.x,
                          msg.pose.pose.orientation.y,
                          msg.pose.pose.orientation.z,
                          msg.pose.pose.orientation.w]]).as_rotvec()[0][2]

        self.configuration[self.hparams.theta_idx] = theta
    
    def get_latest_configuration(self):
        return self.configuration
    
    def set_desired_target_position_request(self, request):
        self.target_position[self.hparams.x_idx] = request.x
        self.target_position[self.hparams.y_idx] = request.y
        
        rospy.loginfo(f"Desired target position successfully set: {self.target_position}")
        return my_tiago_msgs.srv.SetDesiredTargetPositionResponse(True)

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

    # Setup loggers for bagfiles
    log = False
    bag_dir = '/tmp/crowd_navigation_tiago/bagfiles'
    if not os.path.exists(bag_dir):
        os.makedirs(bag_dir)

    if log:
        odom_topic = "/mobile_base_controller/odom"
        cmd_vel_topic = "/mobile_base_controller/cmd_vel"
        odom_bagname = "odometry.bag"
        cmd_vel_bagname = "commands.bag"
    
        odom_bag = os.path.join(bag_dir, odom_bagname)
        cmd_vel_bag = os.path.join(bag_dir, cmd_vel_bagname)

        odom_logger = Logger(odom_topic, odom_bag)
        cmd_vel_logger = Logger(cmd_vel_topic, cmd_vel_bag)

        # Start loggers
        odom_logger.start_logging()
        cmd_vel_logger.start_logging()

    # Setup initial configuration
    rospy.loginfo("Waiting for the first message on topic %s" % controller_manager.odom_topic)
    rospy.wait_for_message(controller_manager.odom_topic, nav_msgs.msg.Odometry)
    starting_configuration = controller_manager.get_latest_configuration()
    controller_manager.init()
    
    print("Init configuration ------------")
    print(starting_configuration)

    try:
        while not(rospy.is_shutdown()):
            controller_manager.update()
            controller_manager.publish_command()
            rate.sleep()
    except rospy.ROSInterruptException as e:
        rospy.logwarn("ROS node shutting down")
        rospy.logwarn('{}'.format(e))
    finally:
        # print(controller_manager.nmpc_controller.max_time)
        if log:
            # Stop loggers
            odom_logger.stop_logging()
            cmd_vel_logger.stop_logging()