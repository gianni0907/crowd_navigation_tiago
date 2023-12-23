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

        self.state = State(0.0, 0.0, 0.0, 0.0, 0.0)
        self.wheels_vel = np.zeros(2) # [w_r, w_l]
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

        # Setup subscriber for joint_states topic
        state_topic = '/joint_states'
        rospy.Subscriber(
            state_topic,
            sensor_msgs.msg.JointState,
            self.joint_states_callback
        )
        
        # Setup subscriber for crowd motion prediction:
        crowd_prediction_topic = 'crowd_motion_prediction'
        rospy.Subscriber(
            crowd_prediction_topic,
            my_tiago_msgs.msg.CrowdMotionPredictionStamped,
            self.crowd_motion_prediction_stamped_callback
        )

        # Set variables to store data
        if self.hparams.log:
            self.state_history = []
            self.prediction_history = []
            self.wheels_vel_history = []
            self.wheels_acc_history = []
            self.target_history = []
            self.obstacles_history = []
            self.residuals = []

    def init(self):
        # Init robot state
        if self.update_state():
            self.status = Status.READY
            self.target_position = np.array([self.state.x,
                                            self.state.y])
            self.nmpc_controller.init(self.state)

    def publish_command(self):
        """
        The NMPC solver returns wheels accelerations as control input
        Transform it into the avilable robot control input: driving and steering velocities
        """
        if self.status == Status.MOVING:
            dt = self.hparams.dt
            alpha_r = self.control_input[self.hparams.r_wheel_idx]
            alpha_l = self.control_input[self.hparams.l_wheel_idx]
            wheel_radius = self.hparams.wheel_radius
            wheel_separation = self.hparams.wheel_separation

            # Compute driving and steering accelerations given inputs
            v_dot = wheel_radius * 0.5 * (alpha_r + alpha_l)
            omega_dot = (wheel_radius / wheel_separation) * (alpha_r - alpha_l)
            
            # Integrate to get the new wheels velocity
            v = self.state.v + v_dot * dt
            omega = self.state.omega + omega_dot * dt
        else:
            v = 0.0
            omega = 0.0

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
    
    def joint_states_callback(self, msg):
        self.wheels_vel = np.array([msg.velocity[13], msg.velocity[12]])

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
        self.state.theta = theta + self.k * 2 * math.pi
        self.state.x = transform.transform.translation.x + self.hparams.b * math.cos(self.state.theta)
        self.state.y = transform.transform.translation.y + self.hparams.b * math.sin(self.state.theta)

    def update_state(self):
        try:
            # Update [x, y, theta]
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_footprint_frame, rospy.Time()
            )
            self.set_from_tf_transform(transform)
            # Update [v, omega]
            self.state.v = self.hparams.wheel_radius * 0.5 * \
                (self.wheels_vel[self.hparams.r_wheel_idx] + self.wheels_vel[self.hparams.l_wheel_idx])
            
            self.state.omega = (self.hparams.wheel_radius / self.hparams.wheel_separation) * \
                (self.wheels_vel[self.hparams.r_wheel_idx] - self.wheels_vel[self.hparams.l_wheel_idx])
            return True
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing current state")
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
        output_dict['states'] = self.state_history
        output_dict['predictions'] = self.prediction_history
        output_dict['wheels_velocities'] = self.wheels_vel_history
        output_dict['wheels_accelerations'] = self.wheels_acc_history
        output_dict['targets'] = self.target_history

        output_dict['n_obstacles'] = self.hparams.n_obstacles
        if self.hparams.n_obstacles > 0:        
            output_dict['obstacles_position'] = self.obstacles_history

        output_dict['x_bounds'] = [self.hparams.x_lower_bound, self.hparams.x_upper_bound]
        output_dict['y_bounds'] = [self.hparams.y_lower_bound, self.hparams.y_upper_bound]
        output_dict['input_bounds'] = [self.hparams.alpha_min, self.hparams.alpha_max]
        output_dict['v_bounds'] = [self.hparams.driving_vel_min, self.hparams.driving_vel_max]
        output_dict['omega_bounds'] = [self.hparams.steering_vel_max_neg, self.hparams.steering_vel_max]
        output_dict['wheels_vel_bounds'] = [self.hparams.w_max_neg, self.hparams.w_max]

        output_dict['rho_cbf'] = self.hparams.rho_cbf
        output_dict['ds_cbf'] = self.hparams.ds_cbf
        output_dict['gamma_cbf'] = self.hparams.gamma_cbf
        output_dict['frequency'] = self.hparams.controller_frequency
        output_dict['N_horizon'] = self.hparams.N_horizon
        output_dict['position_weight'] = self.hparams.p_weight
        output_dict['v_weight'] = self.hparams.v_weight
        output_dict['omega_weight'] = self.hparams.omega_weight
        output_dict['input_weight'] = self.hparams.u_weight
        output_dict['terminal_factor'] = self.hparams.terminal_factor
        output_dict['offset_b'] = self.hparams.b
        output_dict['residuals'] = self.residuals
        
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
            q_ref[:self.hparams.y_idx + 1, k] = self.target_position
        u_ref = np.zeros((self.nmpc_controller.nu, self.hparams.N_horizon))
        q_ref[:self.hparams.y_idx + 1, self.hparams.N_horizon] = self.target_position
        
        if self.hparams.n_obstacles > 0:
            if self.data_lock.acquire(False):
                self.crowd_motion_prediction_stamped_rt = self.crowd_motion_prediction_stamped
                self.data_lock.release()

        self.data_lock.acquire()
        flag = self.update_state()
        self.data_lock.release()
        
        if flag and (self.sensing or self.hparams.n_obstacles == 0) and self.status == Status.MOVING:
            # Compute the position error
            error = np.array([self.target_position[self.hparams.x_idx] - self.state.x, 
                              self.target_position[self.hparams.y_idx] - self.state.y])
            
            if norm(error) < self.hparams.error_tol:
                self.control_input = np.zeros((self.nmpc_controller.nu))
                print("Stop state #######################")
                print(self.state)
                print("##########################################")
                self.status = Status.READY
            else:
                try:
                    self.nmpc_controller.update(
                        self.state,
                        q_ref,
                        u_ref
                    )
                    self.control_input = self.nmpc_controller.get_command()
                except Exception as e:
                    rospy.logwarn("NMPC solver failed")
                    rospy.logwarn('{}'.format(e))
                    self.control_input = np.zeros((self.nmpc_controller.nu))
                    print("Stop state #######################")
                    print(self.state)
                    print("##########################################")
                    self.status = Status.READY
                
        else:
            if not(self.sensing) and self.hparams.n_obstacles > 0:
                rospy.logwarn("Missing sensing info")
            self.control_input = np.zeros((self.nmpc_controller.nu))
            if self.status == Status.MOVING:
                print("Stop state #######################")
                print(self.state)
                print("##########################################")
                self.status = Status.READY
            
    def run(self):
        rate = rospy.Rate(self.hparams.controller_frequency)
        # Variables for Analysis of required time
        self.previous_time = rospy.get_time()

        # Waiting for initial state
        while self.status == Status.WAITING:
            self.init()
        print("Initial state ********************")
        print(self.state)
        print("******************************************")

        try:
            while not(rospy.is_shutdown()):
                time = rospy.get_time()
                init_time = time
 
                self.update()
                self.publish_command()
                
                # Saving data for plots
                if self.hparams.log and (self.sensing or self.hparams.n_obstacles == 0):
                    self.state_history.append([
                        self.state.x,
                        self.state.y,
                        self.state.theta,
                        self.state.v,
                        self.state.omega,
                        time
                    ])
                    self.wheels_vel_history.append([
                        self.wheels_vel[self.hparams.r_wheel_idx],
                        self.wheels_vel[self.hparams.l_wheel_idx],
                        time
                    ])
                    self.wheels_acc_history.append([
                        self.control_input[self.hparams.r_wheel_idx],
                        self.control_input[self.hparams.l_wheel_idx],
                        time
                    ])
                    self.target_history.append([
                        self.target_position[self.hparams.x_idx],
                        self.target_position[self.hparams.y_idx],
                        time
                    ])
                    predicted_trajectory = np.zeros((self.nmpc_controller.nq, self.hparams.N_horizon+2))
                    for i in range(self.hparams.N_horizon):
                        predicted_trajectory[:, i] = self.nmpc_controller.acados_ocp_solver.get(i,'x')
                    predicted_trajectory[:, self.hparams.N_horizon] = \
                        self.nmpc_controller.acados_ocp_solver.get(self.hparams.N_horizon, 'x')
                    predicted_trajectory[:, self.hparams.N_horizon + 1] = time

                    self.prediction_history.append(predicted_trajectory.tolist())
                    self.residuals.append(self.nmpc_controller.acados_ocp_solver.get_stats('residuals').tolist())
                    
                    if self.hparams.n_obstacles > 0:
                        first_predictions = []
                        for i in range(self.hparams.n_obstacles):
                            first_predictions.append([
                                self.crowd_motion_prediction_stamped_rt.crowd_motion_prediction.motion_predictions[i].positions[0].x,
                                self.crowd_motion_prediction_stamped_rt.crowd_motion_prediction.motion_predictions[i].positions[0].y,
                                time
                            ])
                        self.obstacles_history.append(first_predictions)

                final_time = rospy.get_time()        
                deltat = final_time - init_time
                if deltat > 1 / (2 * self.hparams.controller_frequency):
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
