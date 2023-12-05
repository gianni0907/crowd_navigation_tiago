import rospy
import nav_msgs.msg
import geometry_msgs.msg
import tf2_ros

from scipy.spatial.transform import Rotation as R

from my_tiago_controller.Hparams import *
from my_tiago_controller.KinematicModel import *
from my_tiago_controller.NMPC import *
from my_tiago_controller.Logger import *
from my_tiago_controller.Status import *

import my_tiago_msgs.srv

class ControllerManager:
    def __init__(
            self,
            controller_frequency,
            nmpc_N,
            nmpc_T):

        self.controller_frequency = controller_frequency
        # Set status
        self.status = Status.WAITING
        
        # NMPC:
        self.dt = 1.0 / self.controller_frequency
        self.hparams = Hparams()
        self.nmpc_controller = NMPC(nmpc_N, nmpc_T)

        self.configuration = np.zeros((self.nmpc_controller.nq))

        # Setup publisher for wheel velocity commands:
        cmd_vel_topic = '/mobile_base_controller/cmd_vel'
        self.cmd_vel_publisher = rospy.Publisher(
            cmd_vel_topic,
            geometry_msgs.msg.Twist,
            queue_size=1
        )

        # Setup reference frames:
        if self.hparams.real_robot:
            self.map_frame = 'map'
        else:
            self.map_frame = 'odom'
        self.base_footprint_frame = 'base_footprint'

        # Setup TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)        

        # Setup ROS Service to set target position:
        self.set_desired_target_position_srv = rospy.Service(
            'SetDesiredTargetPosition',
            my_tiago_msgs.srv.SetDesiredTargetPosition,
            self.set_desired_target_position_request
        )

    def init(self):
        # Init robot configuration
        self.update_configuration()
        if self.status == Status.READY:
            self.target_position = np.array([self.configuration[self.hparams.x_idx],
                                            self.configuration[self.hparams.y_idx]])
            self.nmpc_controller.init(self.configuration)

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

    def set_from_tf_transform(self, transform):
        self.configuration[self.hparams.x_idx] = transform.transform.translation.x
        self.configuration[self.hparams.y_idx] = transform.transform.translation.y
        q = transform.transform.rotation
        self.configuration[self.hparams.theta_idx] = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def update_configuration(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_footprint_frame, rospy.Time()
            )
            self.set_from_tf_transform(transform)
            self.status = Status.READY
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            self.status = Status.WAITING

    def set_desired_target_position_request(self, request):
        if self.status != Status.READY:
            rospy.loginfo("Cannot set desired target position, robot is not READY")
            return my_tiago_msgs.srv.SetDesiredTargetPositionResponse(False)
        
        self.target_position[self.hparams.x_idx] = request.x
        self.target_position[self.hparams.y_idx] = request.y
        
        rospy.loginfo(f"Desired target position successfully set: {self.target_position}")
        return my_tiago_msgs.srv.SetDesiredTargetPositionResponse(True)

    def update(self):
        q_ref = np.zeros((self.nmpc_controller.nq, self.nmpc_controller.N+1))
        for k in range(self.nmpc_controller.N):
            q_ref[:self.nmpc_controller.nq - 1, k] = self.target_position
        u_ref = np.zeros((self.nmpc_controller.nu, self.nmpc_controller.N))
        q_ref[:self.nmpc_controller.nq - 1, self.nmpc_controller.N] = self.target_position
        self.update_configuration()

        if self.status == Status.READY:
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
                self.control_input = np.zeros((self.nmpc_controller.nu))
        else:
            self.control_input = np.zeros((self.nmpc_controller.nu))
        

def main():
    rospy.init_node('tiago_nmpc_controller', log_level=rospy.INFO)
    rospy.loginfo('TIAGo control module [OK]')

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
    rate = rospy.Rate(controller_frequency)

    # Setup loggers for bagfiles
    bag_dir = '/tmp/crowd_navigation_tiago/bagfiles'
    if not os.path.exists(bag_dir):
        os.makedirs(bag_dir)

    if controller_manager.hparams.log:
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

    # Waiting for current configuration to initialize controller_manager
    while controller_manager.status == Status.WAITING:
        controller_manager.init()
    starting_configuration = controller_manager.configuration
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
        if controller_manager.hparams.log:
            # Stop loggers
            odom_logger.stop_logging()
            cmd_vel_logger.stop_logging()