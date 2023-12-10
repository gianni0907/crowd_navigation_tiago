import json
import rospy
import math
import os
import geometry_msgs.msg
import tf2_ros

from numpy.linalg import *

from my_tiago_controller.Hparams import *
from my_tiago_controller.NMPC import *
from my_tiago_controller.Status import *
from my_tiago_controller.utils import *

import my_tiago_msgs.srv

class ControllerManager:
    def __init__(
            self,
            hparams : Hparams):

        self.controller_frequency = hparams.controller_frequency
        
        # Set status
        self.status = Status.WAITING

        # NMPC:
        self.hparams = hparams
        self.nmpc_controller = NMPC(hparams)

        self.configuration = Configuration(0.0, 0.0, 0.0)

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

        # Setup ROS Service to set humans trajectories:
        self.set_humans_trajectory_srv = rospy.Service(
            'SetHumansTrajectory',
            my_tiago_msgs.srv.SetHumansTrajectory,
            self.set_humans_trajectory_request
        )

        # Set variables to store data
        if self.hparams.log:
            self.configuration_history = []
            self.control_input_history = []
            self.prediction_history = []
            self.velocity_history = []
            self.target_history = []
            self.humans_history = []

    def init(self):
        # Init robot configuration
        
        if self.update_configuration():
            self.status = Status.READY
            self.target_position = np.array([self.configuration.x,
                                            self.configuration.y])
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
        return v, omega

    def set_from_tf_transform(self, transform):
        self.configuration.x = transform.transform.translation.x
        self.configuration.y = transform.transform.translation.y
        q = transform.transform.rotation
        self.configuration.theta = math.atan2(
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
        
    def set_humans_trajectory_request(self, request):
        if self.hparams.dynamic:
            rospy.loginfo("Cannot set humans trajectory, humans are already moving")
            return my_tiago_msgs.srv.SetDesiredTargetPositionResponse(False)
        else:
            self.hparams.trajectories[0, :, 0] = request.x_1
            self.hparams.trajectories[0, :, 1] = request.y_1
            self.hparams.trajectories[1, :, 0] = request.x_2
            self.hparams.trajectories[1, :, 1] = request.y_2
            self.hparams.trajectories[2, :, 0] = request.x_3
            self.hparams.trajectories[2, :, 1] = request.y_3
            self.hparams.trajectories[3, :, 0] = request.x_4
            self.hparams.trajectories[3, :, 1] = request.y_4
            self.hparams.trajectories[4, :, 0] = request.x_5
            self.hparams.trajectories[4, :, 1] = request.y_5
            self.hparams.dynamic = True
            rospy.loginfo("Humans trajectory successfully set")
            return my_tiago_msgs.srv.SetHumansTrajectoryResponse(True)

    def log_values(self, filename):
        output_dict = {}
        output_dict['configurations'] = self.configuration_history
        output_dict['inputs'] = self.control_input_history
        output_dict['predictions'] = self.prediction_history
        output_dict['velocities'] = self.velocity_history
        output_dict['targets'] = self.target_history
        output_dict['x_bounds'] = [self.hparams.x_lower_bound, self.hparams.x_upper_bound]
        output_dict['y_bounds'] = [self.hparams.y_lower_bound, self.hparams.y_upper_bound]
        output_dict['control_bounds'] = [self.hparams.w_max_neg, self.hparams.w_max]
        output_dict['v_bounds'] = [self.hparams.driving_vel_min, self.hparams.driving_vel_max]
        output_dict['omega_bounds'] = [self.hparams.steering_vel_max_neg, self.hparams.steering_vel_max]
        output_dict['humans_position'] = self.humans_history
        output_dict['rho_cbf'] = self.hparams.rho_cbf
        output_dict['ds_cbf'] = self.hparams.ds_cbf
        output_dict['gamma_cbf'] = self.hparams.gamma_cbf
        output_dict['frequency'] = self.controller_frequency
        output_dict['N_horizon'] = self.hparams.N_horizon
        output_dict['Q_mat_weights'] = self.hparams.q
        output_dict['R_mat_weights'] = self.hparams.r
        output_dict['terminal_factor'] = self.hparams.q_factor
        
        # log the data in a .json file
        log_dir = '/tmp/crowd_navigation_tiago/data'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, filename)
        with open(log_path, 'w') as file:
            json.dump(output_dict, file)

    def update(self):
        q_ref = np.zeros((self.nmpc_controller.nq, self.hparams.N_horizon+1))
        for k in range(self.hparams.N_horizon):
            q_ref[:self.nmpc_controller.nq - 1, k] = self.target_position
        u_ref = np.zeros((self.nmpc_controller.nu, self.hparams.N_horizon))
        q_ref[:self.nmpc_controller.nq - 1, self.hparams.N_horizon] = self.target_position
        
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
    hparams = Hparams()
    controller_manager = ControllerManager(hparams)
    rate = rospy.Rate(hparams.controller_frequency)
    N_horizon = hparams.N_horizon

    # Waiting for current configuration to initialize controller_manager
    while controller_manager.status == Status.WAITING:
        controller_manager.init()
    print("Initial configuration ********************")
    print(controller_manager.configuration)
    print("******************************************")

    try:
        while not(rospy.is_shutdown()):
            time = rospy.get_time()

            controller_manager.update()
            v, omega = controller_manager.publish_command()

            # Saving data for plots
            if hparams.log:
                controller_manager.configuration_history.append([
                    controller_manager.configuration.x,
                    controller_manager.configuration.y,
                    controller_manager.configuration.theta,
                    time
                ])
                controller_manager.control_input_history.append([
                    controller_manager.control_input[hparams.wr_idx],
                    controller_manager.control_input[hparams.wl_idx],
                    time
                ])
                controller_manager.velocity_history.append([v, omega, time])
                controller_manager.target_history.append([
                    controller_manager.target_position[hparams.x_idx],
                    controller_manager.target_position[hparams.y_idx],
                    time
                ])
                predicted_trajectory = np.zeros((controller_manager.nmpc_controller.nq, N_horizon+1))
                for i in range(N_horizon):
                    predicted_trajectory[:, i] = \
                        controller_manager.nmpc_controller.acados_ocp_solver.get(i,'x')
                predicted_trajectory[:, N_horizon] = \
                    controller_manager.nmpc_controller.acados_ocp_solver.get(N_horizon, 'x')
                controller_manager.prediction_history.append(predicted_trajectory.tolist())

                controller_manager.humans_history.append(hparams.trajectories[:, hparams.traj_iter, :].tolist())

            if hparams.dynamic:
                hparams.traj_iter = hparams.traj_iter + 1
                if hparams.traj_iter == hparams.n_traj_steps * 2:
                    hparams.traj_iter = 0
                    hparams.dynamic = False
        
            # Checking the position error
            error = np.array([controller_manager.target_position[hparams.x_idx] - \
                              controller_manager.configuration.x,
                              controller_manager.target_position[hparams.y_idx] - \
                              controller_manager.configuration.y])
            if norm(error) < hparams.error_tol and controller_manager.status == Status.MOVING:
                    print("Stop configuration #######################")
                    print(controller_manager.configuration)
                    print("##########################################")
                    controller_manager.status = Status.READY
            rate.sleep()
    except rospy.ROSInterruptException as e:
        rospy.logwarn("ROS node shutting down")
        rospy.logwarn('{}'.format(e))
    finally:
        print(f"Maximum required time to find NMPC solution is {controller_manager.nmpc_controller.max_time} " \
              f"at instant {controller_manager.nmpc_controller.idx_time}")
        controller_manager.log_values(hparams.logfile)