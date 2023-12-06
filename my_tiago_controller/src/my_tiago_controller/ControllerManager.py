import json
import rospy
import geometry_msgs.msg
import tf2_ros

from numpy.linalg import *

from my_tiago_controller.Hparams import *
from my_tiago_controller.KinematicModel import *
from my_tiago_controller.NMPC import *
from my_tiago_controller.Status import *
from my_tiago_controller.utils import *

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

        # Set variables to store data
        if self.hparams.log:
            self.configuration_history = []
            self.prediction_history = []
            self.control_input_history = []

    def init(self):
        # Init robot configuration
        
        if self.update_configuration():
            self.status = Status.READY
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
            return True
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing current configuration")
            return False

    def set_desired_target_position_request(self, request):
        if self.status == Status.WAITING:
            rospy.loginfo("Cannot set desired target position, robot is not READY")
            return my_tiago_msgs.srv.SetDesiredTargetPositionResponse(False)
        else:
            self.target_position[self.hparams.x_idx] = request.x
            self.target_position[self.hparams.y_idx] = request.y
            if self.status == Status.READY:
                self.status = Status.MOVING        
                rospy.loginfo(f"Desired target position successfully set: {self.target_position}")
            elif self.status == Status.MOVING:
                rospy.loginfo(f"Desired target position successfully changed: {self.target_position}")
            return my_tiago_msgs.srv.SetDesiredTargetPositionResponse(True)

    def log_values(self, filename):
        output_dict = {}
        output_dict['configurations'] = self.configuration_history
        output_dict['inputs'] = self.control_input_history
        output_dict['predictions'] = self.prediction_history
        output_dict['x_bounds'] = [self.hparams.x_lower_bound, self.hparams.x_upper_bound]
        output_dict['y_bounds'] = [self.hparams.y_lower_bound, self.hparams.y_upper_bound]
        output_dict['control_bounds'] = [self.hparams.w_max_neg, self.hparams.w_max]
        # log the data in a .json file
        log_dir = '/tmp/crowd_navigation_tiago/data'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, filename)
        with open(log_path, 'w') as file:
            json.dump(output_dict, file)

    def update(self):
        q_ref = np.zeros((self.nmpc_controller.nq, self.nmpc_controller.N+1))
        for k in range(self.nmpc_controller.N):
            q_ref[:self.nmpc_controller.nq - 1, k] = self.target_position
        u_ref = np.zeros((self.nmpc_controller.nu, self.nmpc_controller.N))
        q_ref[:self.nmpc_controller.nq - 1, self.nmpc_controller.N] = self.target_position
        
        self.update_configuration()

        if self.status == Status.MOVING:
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

    # Waiting for current configuration to initialize controller_manager
    while controller_manager.status == Status.WAITING:
        controller_manager.init()
    print("Init configuration ------------")
    print(controller_manager.configuration)

    try:
        while not(rospy.is_shutdown()):
            time = rospy.get_time()

            controller_manager.update()
            controller_manager.publish_command()

            # Saving data for plots
            if controller_manager.hparams.log:
                controller_manager.configuration_history.append([
                    controller_manager.configuration[controller_manager.hparams.x_idx],
                    controller_manager.configuration[controller_manager.hparams.y_idx],
                    controller_manager.configuration[controller_manager.hparams.theta_idx],
                    time
                ])
                controller_manager.control_input_history.append([
                    controller_manager.control_input[controller_manager.hparams.wr_idx],
                    controller_manager.control_input[controller_manager.hparams.wl_idx],
                    time
                ])
                predicted_trajectory = np.zeros((controller_manager.nmpc_controller.nq, N_horizon+1))
                for i in range(N_horizon):
                    predicted_trajectory[:, i] = \
                        controller_manager.nmpc_controller.acados_ocp_solver.get(i,'x')
                predicted_trajectory[:, N_horizon] = \
                    controller_manager.nmpc_controller.acados_ocp_solver.get(N_horizon, 'x')
                controller_manager.prediction_history.append(predicted_trajectory.tolist())

            # Checking the position error
            error = np.array([controller_manager.target_position[controller_manager.hparams.x_idx] - \
                              controller_manager.configuration[controller_manager.hparams.x_idx],
                              controller_manager.target_position[controller_manager.hparams.y_idx] - \
                              controller_manager.configuration[controller_manager.hparams.y_idx]])
            if norm(error) < controller_manager.hparams.error_tol:
                controller_manager.status = Status.READY
            rate.sleep()
    except rospy.ROSInterruptException as e:
        rospy.logwarn("ROS node shutting down")
        rospy.logwarn('{}'.format(e))
    finally:
        # print(controller_manager.nmpc_controller.max_time)
        controller_manager.log_values(controller_manager.hparams.logfile)