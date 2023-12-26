import json
import rospy
import threading
import math
import os
import geometry_msgs.msg
import sensor_msgs.msg
import tf2_ros

from numpy.linalg import *

from my_tiago_controller.Hparams import *
from my_tiago_controller.NMPC import *
from my_tiago_controller.Status import *
from my_tiago_controller.utils import *

import my_tiago_msgs.srv

class ControllerManager:
    def __init__(self):
        # Set the Hyperparameters
        self.hparams = Hparams()
        
        # Set status
        self.status = Status.WAITING # WAITING for the initial robot configuration
        self.sensing = False

        # counter for the angle unwrapping
        self.k = 0
        self.previous_theta = 0.0
        
        # NMPC:
        self.nmpc_controller = NMPC(self.hparams)

        self.configuration = Configuration(0.0, 0.0, 0.0)
        self.data_lock = threading.Lock()
        self.crowd_motion_prediction_stamped = CrowdMotionPredictionStamped(rospy.Time.now(),
                                                                            'map',
                                                                            CrowdMotionPrediction())
        # Set real-time prediction:
        self.crowd_motion_prediction_stamped_rt = self.crowd_motion_prediction_stamped

        # Setup publisher for wheel velocity commands:
        cmd_vel_topic = '/mobile_base_controller/cmd_vel'
        self.cmd_vel_publisher = rospy.Publisher(
            cmd_vel_topic,
            geometry_msgs.msg.Twist,
            queue_size=1
        )

        # Setup reference frames:
        self.map_frame = 'map'
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

        # # Setup subscriber for joint_states topic
        # state_topic = '/joint_states'
        # rospy.Subscriber(
        #     state_topic,
        #     sensor_msgs.msg.JointState,
        #     self.joint_states_callback
        # )
        
        # Setup subscriber for crowd motion prediction:
        crowd_prediction_topic = 'crowd_motion_prediction'
        rospy.Subscriber(
            crowd_prediction_topic,
            my_tiago_msgs.msg.CrowdMotionPredictionStamped,
            self.crowd_motion_prediction_stamped_callback
        )

        # Set variables to store data
        if self.hparams.log:
            self.configuration_history = []
            self.control_input_history = []
            self.prediction_history = []
            self.velocity_history = []
            self.target_history = []
            self.humans_history = []
            # self.robot_velocity_history = []

    def init(self):
        # Init robot configuration
        if self.update_configuration():
            self.status = Status.READY
            # Initialize target position to the current position
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
    
    # def joint_states_callback(self, msg):
    #     w_l = msg.velocity[12]
    #     w_r = msg.velocity[13]
    #     self.robot_velocity_history.append([w_r, w_l, rospy.get_time()])

    def crowd_motion_prediction_stamped_callback(self, msg):
        if not self.sensing:
            self.sensing = True
        crowd_motion_prediction_stamped = CrowdMotionPredictionStamped.from_message(msg)
        self.data_lock.acquire()
        self.crowd_motion_prediction_stamped = crowd_motion_prediction_stamped
        self.data_lock.release()

    def set_from_tf_transform(self, transform):
        q = transform.transform.rotation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        product = self.previous_theta * theta
        if product < - 8:
            # passing through pi [rad]
            if theta > 0.0:
                # from negative angle to positive angle
                self.k -= 1
            elif theta < 0.0:
                # from positive angle to negative angle
                self.k += 1
        self.previous_theta = theta
        self.configuration.theta = theta + self.k * 2 * math.pi
        self.configuration.x = transform.transform.translation.x + self.hparams.b * casadi.cos(self.configuration.theta)
        self.configuration.y = transform.transform.translation.y + self.hparams.b * casadi.sin(self.configuration.theta)

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
        output_dict['velocities'] = self.velocity_history
        # output_dict['robot_velocities'] = self.robot_velocity_history
        output_dict['targets'] = self.target_history
        output_dict['x_bounds'] = [self.hparams.x_lower_bound, self.hparams.x_upper_bound]
        output_dict['y_bounds'] = [self.hparams.y_lower_bound, self.hparams.y_upper_bound]
        output_dict['control_bounds'] = [self.hparams.w_max_neg, self.hparams.w_max]
        output_dict['v_bounds'] = [self.hparams.driving_vel_min, self.hparams.driving_vel_max]
        output_dict['omega_bounds'] = [self.hparams.steering_vel_max_neg, self.hparams.steering_vel_max]
        output_dict['n_obstacles'] = self.hparams.n_obstacles
        if self.hparams.n_obstacles > 0:        
            output_dict['humans_position'] = self.humans_history
        output_dict['rho_cbf'] = self.hparams.rho_cbf
        output_dict['ds_cbf'] = self.hparams.ds_cbf
        output_dict['gamma_cbf'] = self.hparams.gamma_cbf
        output_dict['frequency'] = self.hparams.controller_frequency
        output_dict['N_horizon'] = self.hparams.N_horizon
        output_dict['Q_mat_weights'] = self.hparams.q
        output_dict['R_mat_weights'] = self.hparams.r
        output_dict['terminal_factor'] = self.hparams.q_factor
        output_dict['offset_b'] = self.hparams.b
        
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
        
        if self.hparams.n_obstacles > 0:
            if self.data_lock.acquire(False):
                self.crowd_motion_prediction_stamped_rt = self.crowd_motion_prediction_stamped
                self.data_lock.release()

        self.data_lock.acquire()
        flag = self.update_configuration()
        # print(self.configuration)
        self.data_lock.release()
        
        if flag and (self.sensing or self.hparams.n_obstacles == 0):
            # Compute the position error
            error = np.array([self.target_position[self.hparams.x_idx] - \
                            self.configuration.x,
                            self.target_position[self.hparams.y_idx] - \
                            self.configuration.y])
            
            if norm(error) < self.hparams.error_tol:
                self.control_input = np.zeros((self.nmpc_controller.nu))
                if self.status == Status.MOVING:
                    print("Stop configuration #######################")
                    print(self.configuration)
                    print("##########################################")
                    self.status = Status.READY
            else:
                try:
                    self.nmpc_controller.update(
                        self.configuration,
                        q_ref,
                        u_ref,
                        self.crowd_motion_prediction_stamped_rt.crowd_motion_prediction
                    )
                    self.control_input = self.nmpc_controller.get_command()
                except Exception as e:
                    rospy.logwarn("NMPC solver failed")
                    rospy.logwarn('{}'.format(e))
                    self.control_input = np.zeros((self.nmpc_controller.nu))
                
        else:
            if not(self.sensing) and self.hparams.n_obstacles > 0:
                rospy.logwarn("Missing sensing info")
            self.control_input = np.zeros((self.nmpc_controller.nu))        
            
    def run(self):
        rate = rospy.Rate(self.hparams.controller_frequency)
        # Variables for Analysis of required time
        self.previous_time = rospy.get_time()

        # Waiting for initial configuration from tf
        while self.status == Status.WAITING:
            self.init()
        print("Initial configuration ********************")
        print(self.configuration)
        print("******************************************")

        try:
            while not(rospy.is_shutdown()):
                time = rospy.get_time()
                init_time = time
 
                self.update()
                v, omega = self.publish_command()
                
                # Saving data for plots
                if self.hparams.log and (self.sensing or self.hparams.n_obstacles == 0):
                    self.configuration_history.append([
                        self.configuration.x,
                        self.configuration.y,
                        self.configuration.theta,
                        time
                    ])
                    self.control_input_history.append([
                        self.control_input[self.hparams.wr_idx],
                        self.control_input[self.hparams.wl_idx],
                        time
                    ])
                    self.velocity_history.append([v, omega, time])
                    self.target_history.append([
                        self.target_position[self.hparams.x_idx],
                        self.target_position[self.hparams.y_idx],
                        time
                    ])
                    predicted_trajectory = np.zeros((self.nmpc_controller.nq, self.hparams.N_horizon+1))
                    for i in range(self.hparams.N_horizon):
                        predicted_trajectory[:, i] = self.nmpc_controller.acados_ocp_solver.get(i,'x')
                    predicted_trajectory[:, self.hparams.N_horizon] = \
                        self.nmpc_controller.acados_ocp_solver.get(self.hparams.N_horizon, 'x')

                    self.prediction_history.append(predicted_trajectory.tolist())
                    
                    if self.hparams.n_obstacles > 0:
                        first_predictions = []
                        for i in range(self.hparams.n_obstacles):
                            first_predictions.append([
                                self.crowd_motion_prediction_stamped_rt.crowd_motion_prediction.motion_predictions[i].positions[0].x,
                                self.crowd_motion_prediction_stamped_rt.crowd_motion_prediction.motion_predictions[i].positions[0].y,
                                time
                            ])
                        self.humans_history.append(first_predictions)

                final_time = rospy.get_time()        
                deltat = final_time - init_time
                if deltat > 1/(2*self.hparams.controller_frequency):
                    print(f"Iteration time {deltat} at instant {time}")

                rate.sleep()
        except rospy.ROSInterruptException as e:
            rospy.logwarn("ROS node shutting down")
            rospy.logwarn('{}'.format(e))
        finally:
            if self.hparams.log:
                self.log_values(self.hparams.logfile)


def main():
    rospy.init_node('tiago_nmpc_controller', log_level=rospy.INFO)
    rospy.loginfo('TIAGo control module [OK]')

    # Build and run controller manager
    controller_manager = ControllerManager()
    controller_manager.run()
