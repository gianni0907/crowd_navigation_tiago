import numpy as np
import math
import time
import os
import json
import threading
import cProfile
import tf2_ros
import rospy
from scipy.spatial.transform import Rotation as R

from crowd_navigation_core.Hparams import *
from crowd_navigation_core.NMPC import *
from crowd_navigation_core.Planner import *
from crowd_navigation_core.utils import *

import gazebo_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import std_msgs.msg
import trajectory_msgs.msg
import control_msgs.msg
import crowd_navigation_msgs.srv

class MotionGenerationManager:
    '''
    Given the crowd prediction, generate the collision-free robot motion
    via NMPC algorithm and CBF-based collision avoidance constraints.
    '''
    def __init__(self):
        self.data_lock = threading.Lock()

        self.hparams = Hparams()
        
        # Set status
        # 3 possibilities:
        #   WAITING for the initial robot state
        #   READY to get the target position while the robot is at rest
        #   MOVING towards the desired target position (the target position can be changed while moving)
        self.status = Status.WAITING

        # Set counter for the angle unwrapping
        self.k = 0
        self.previous_theta = 0.0

        # Planner
        self.planner = Planner(self.hparams.areas_index, self.hparams.intersections, self.hparams.viapoints)

        # NMPC:
        self.nmpc_controller = NMPC(self.hparams)

        self.state = State(0.0, 0.0, 0.0, 0.0, 0.0)
        self.v = 0.0
        self.omega = 0.0
        self.wheels_vel_nonrt = np.zeros(2) # [w_r, w_l]
        if self.hparams.simulation:
            self.agents_pos_nonrt = np.zeros((self.hparams.n_agents, 2))
            self.agents_name = ['actor_{}'.format(i) for i in range(self.hparams.n_agents)]
        self.crowd_motion_prediction_stamped_nonrt = None
        self.crowd_motion_prediction_stamped = CrowdMotionPredictionStamped(rospy.Time.now(),
                                                                            'map',
                                                                            CrowdMotionPrediction())

        # Set variables to store data
        if self.hparams.log:
            self.time_history = []
            self.robot_state_history = []
            self.camera_pos_history = []
            self.camera_horz_angle_history = []
            self.robot_prediction_history = []
            self.wheels_vel_history = []
            self.inputs_history = []
            self.commands_history = []
            self.target_history = []
            self.target_viapoint_history = []
            self.area_index_history = []
            if self.hparams.n_filters > 0:
                self.agents_prediction_history = []
            if self.hparams.simulation:
                self.agents_pos_history = []

        # Setup reference frames:
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'
        self.laser_frame = 'base_laser_link'
        self.camera_frame = 'xtion_rgb_optical_frame'

        # Setup TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Setup subscriber to joint_states topic
        state_topic = '/joint_states'
        rospy.Subscriber(
            state_topic,
            sensor_msgs.msg.JointState,
            self.joint_states_callback
        )
        
        # Setup subscriber to crowd motion prediction topic
        crowd_prediction_topic = 'crowd_motion_prediction'
        rospy.Subscriber(
            crowd_prediction_topic,
            crowd_navigation_msgs.msg.CrowdMotionPredictionStamped,
            self.crowd_motion_prediction_stamped_callback
        )

        # setup subscriber to measurements topic
        self.measurements_stamped_nonrt = None
        meas_topic = 'measurements'
        rospy.Subscriber(
            meas_topic,
            crowd_navigation_msgs.msg.MeasurementsSetStamped,
            self.measurements_callback
        )

        # Setup subscriber to model_states topic
        model_states_topic = "/gazebo/model_states"
        rospy.Subscriber(
            model_states_topic,
            gazebo_msgs.msg.ModelStates,
            self.gazebo_model_states_callback
        )

        # Setup ROS Service to set target position:
        self.set_desired_target_position_srv = rospy.Service(
            'SetDesiredTargetPosition',
            crowd_navigation_msgs.srv.SetDesiredTargetPosition,
            self.set_desired_target_position_request
        )

        # Setup ROS Service to set desired head configuration:
        self.set_desired_head_config_srv = rospy.Service(
            'SetDesiredHeadConfig',
            crowd_navigation_msgs.srv.SetDesiredHeadConfig,
            self.set_desired_head_config_request
        )

        # Setup publisher to cmd_vel topic
        cmd_vel_topic = '/mobile_base_controller/cmd_vel'
        self.cmd_vel_publisher = rospy.Publisher(
            cmd_vel_topic,
            geometry_msgs.msg.Twist,
            queue_size=1
        )

        # Setup publisher to /head_controller/command topic
        head_command_topic = '/head_controller/command'
        self.head_config_publisher = rospy.Publisher(
            head_command_topic,
            trajectory_msgs.msg.JointTrajectory,
            queue_size=1)

        # Setup publisher to /head_controller/point_head_action topic
        point_head_action_topic = '/head_controller/point_head_action/goal'
        self.point_head_action_publisher = rospy.Publisher(
            point_head_action_topic,
            control_msgs.msg.PointHeadActionGoal,
            queue_size=1)

    def init(self):
        # Initialize target position to the current position
        self.target_position = np.array([self.state.x,
                                         self.state.y])
        self.current_target_position = self.target_position
        self.error = np.zeros((4))
        # Initialize the NMPC controller
        self.nmpc_controller.init(self.state)
        self.area_idx = get_area_index(self.hparams.areas,
                                       self.hparams.a_coefs,
                                       self.hparams.b_coefs,
                                       self.hparams.c_coefs,
                                       self.state.get_state()[:2])[0]
        self.next_area_idx = self.area_idx

    def joint_states_callback(self, msg):
        left_wheel_idx = msg.name.index('wheel_left_joint')
        right_wheel_idx = msg.name.index('wheel_right_joint')
        with self.data_lock:
            self.wheels_vel_nonrt = np.array([msg.velocity[right_wheel_idx], msg.velocity[left_wheel_idx]])

    def crowd_motion_prediction_stamped_callback(self, msg):
        with self.data_lock: 
            self.crowd_motion_prediction_stamped_nonrt = CrowdMotionPredictionStamped.from_message(msg)

    def measurements_callback(self, msg):
        with self.data_lock:
            self.measurements_stamped_nonrt = MeasurementsSetStamped.from_message(msg)

    def gazebo_model_states_callback(self, msg):
        if self.hparams.simulation:
            agents_pos = np.zeros((self.hparams.n_agents, 2))
            idx = 0
            for agent_name in self.agents_name:
                if agent_name in msg.name:
                    agent_idx = msg.name.index(agent_name)
                    p = msg.pose[agent_idx].position
                    agent_pos = np.array([p.x, p.y])
                    agents_pos[idx] = agent_pos
                    idx += 1
            with self.data_lock:
                self.agents_pos_nonrt = agents_pos

    def unwrap_theta(self, theta):
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
        return theta + self.k * 2 * math.pi

    def update_state(self, time):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_footprint_frame, time
            )
            robot_config = Configuration.set_from_tf_transform(transform, self.hparams.b)
            # Update state
            self.state.theta = self.unwrap_theta(robot_config.theta)
            self.state.x = robot_config.x
            self.state.y = robot_config.y
            self.state.v = self.v
            self.state.omega = self.omega

            pos = np.array([transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z])
            q = np.array([transform.transform.rotation.x,
                          transform.transform.rotation.y,
                          transform.transform.rotation.z,
                          transform.transform.rotation.w])
            rotation_matrix = R.from_quat(q).as_matrix().T
            self.base_pose = np.eye(4)
            self.base_pose[:3, :3] = rotation_matrix
            self.base_pose[:3, 3] = - np.matmul(rotation_matrix, pos)
            self.state
            return True
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing current state")
            return False
        
    def get_laser_relative_position(self, time):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_footprint_frame, self.laser_frame, time
            )            
            pos = np.array([transform.transform.translation.x - self.hparams.b, 
                            transform.transform.translation.y])
            
            return pos
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing laser pose")
            return None

    def get_camera_pose(self, time):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.camera_frame, time
            )
            q = np.array([transform.transform.rotation.x,
                          transform.transform.rotation.y,
                          transform.transform.rotation.z,
                          transform.transform.rotation.w])
            
            self.camera_horz_angle = math.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                                                1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
            
            pos = np.array([transform.transform.translation.x, 
                            transform.transform.translation.y, 
                            transform.transform.translation.z])
            
            rotation_matrix = R.from_quat(q).as_matrix()
            self.camera_pose = np.eye(4)
            self.camera_pose[:3, :3] = rotation_matrix
            self.camera_pose[:3, 3] = pos
            return True
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing camera pose")
            return False

    def adapt_measurements_format(self, measurements_set_stamped):
        if measurements_set_stamped is not None:
            measurements_set = measurements_set_stamped.measurements_set
            measurements = np.zeros((measurements_set.size, 2))
            for i in range(measurements_set.size):
                measurements[i] = np.array([measurements_set.measurements[i].x,
                                            measurements_set.measurements[i].y])

            return measurements
        else:
            return None
    
    def set_desired_target_position_request(self, request):
        if self.status == Status.WAITING:
            rospy.loginfo("Cannot set desired target position, robot is not READY")
            return crowd_navigation_msgs.srv.SetDesiredTargetPositionResponse(False)
        else:
            self.target_position[self.hparams.x_idx] = request.x
            self.target_position[self.hparams.y_idx] = request.y
            if self.status == Status.READY: 
                self.status = Status.MOVING   
                rospy.loginfo(f"Desired target position successfully set: {self.target_position}")
            elif self.status == Status.MOVING:
                rospy.loginfo(f"Desired target position successfully changed: {self.target_position}")
            target_area_indexes = get_area_index(self.hparams.areas,
                                                 self.hparams.a_coefs,
                                                 self.hparams.b_coefs,
                                                 self.hparams.c_coefs,
                                                 self.target_position)
            if len(target_area_indexes) > 0:
                target_area_idx = target_area_indexes[0]
            else:
                target_area_idx = get_closest_area_index(self.hparams.areas,
                                                         self.target_position)
            _, path = self.planner.compute_path(self.area_idx, target_area_idx)
            print(f'Path={path}')
            self.next_area_idx, self.current_target_position = self.planner.get_next_step()
            if self.current_target_position is None:
                self.current_target_position = self.target_position
            self.point_head(self.adapt_measurements_format(self.measurements_stamped_nonrt))
            return crowd_navigation_msgs.srv.SetDesiredTargetPositionResponse(True)
        
    def set_desired_head_config_request(self, request):
        trajectory = trajectory_msgs.msg.JointTrajectory()

        # Set the header
        trajectory.header = std_msgs.msg.Header()
        trajectory.header.stamp = rospy.Time.now()
        
        # Set the joint names for the head
        trajectory.joint_names = ['head_1_joint', 'head_2_joint']
        
        # Create a JointTrajectoryPoint message
        point = trajectory_msgs.msg.JointTrajectoryPoint()

        # Set the joint positions (pan and tilt)
        point.positions = [request.pan, request.tilt]

        point.velocities = [0.0, 0.0]
        
        # Set the time to reach the target (optional, set to 1 second)
        point.time_from_start = rospy.Duration(1.0)
        
        # Add the point to the trajectory
        trajectory.points = [point]
        
        # Publish the joint state
        self.head_config_publisher.publish(trajectory)
        
        # Return a response indicating success
        rospy.loginfo(f"Desired head configuration successfully set: {point.positions}")
        return crowd_navigation_msgs.srv.SetDesiredHeadConfigResponse(success=True)

    def publish_command(self, control_input):
        """
        The NMPC solver returns wheels accelerations as control input
        Map it into the admissible robot commands: driving and steering velocities
        """
        if all(input == 0.0 for input in control_input):
            self.v = 0.0
            self.omega = 0.0
        else:
            dt = 1 / self.hparams.generator_frequency
            alpha_r = control_input[self.hparams.r_wheel_idx]
            alpha_l = control_input[self.hparams.l_wheel_idx]
            wheel_radius = self.hparams.wheel_radius
            wheel_separation = self.hparams.wheel_separation

            # Compute driving and steering accelerations given inputs
            v_dot = wheel_radius * 0.5 * (alpha_r + alpha_l)
            omega_dot = (wheel_radius / wheel_separation) * (alpha_r - alpha_l)
            
            # Integrate to get the new wheels velocity
            self.v = self.v + v_dot * dt
            self.omega = self.omega + omega_dot * dt

        # Create a twist ROS message:
        cmd_vel_msg = geometry_msgs.msg.Twist()
        cmd_vel_msg.linear.x = self.v
        cmd_vel_msg.linear.y = 0.0
        cmd_vel_msg.linear.z = 0.0
        cmd_vel_msg.angular.x = 0.0
        cmd_vel_msg.angular.y = 0.0
        cmd_vel_msg.angular.z = self.omega

        # Publish wheel velocity commands
        self.cmd_vel_publisher.publish(cmd_vel_msg)
        return self.v, self.omega
        
    def log_values(self):
        output_dict = {}
        output_dict['cpu_time'] = self.time_history
        output_dict['robot_state'] = self.robot_state_history
        output_dict['robot_predictions'] = self.robot_prediction_history
        output_dict['wheels_velocities'] = self.wheels_vel_history
        output_dict['inputs'] = self.inputs_history
        output_dict['commands'] = self.commands_history
        output_dict['targets'] = self.target_history
        output_dict['target_viapoints'] = self.target_viapoint_history
        output_dict['area_index'] = self.area_index_history
        output_dict['n_filters'] = self.hparams.n_filters
        if self.hparams.n_filters > 0:        
            output_dict['agents_predictions'] = self.agents_prediction_history
        output_dict['simulation'] = self.hparams.simulation
        output_dict['perception'] = Perception.print(self.hparams.perception)
        if self.hparams.simulation:
            output_dict['n_agents'] = self.hparams.n_agents
            output_dict['agents_pos'] = self.agents_pos_history
        output_dict['areas'] = [arr.tolist() for arr in self.hparams.areas]
        output_dict['viapoints'] = [point.tolist() for point in self.hparams.viapoints.values() if point is not None]
        output_dict['walls'] = self.hparams.walls
        output_dict['input_bounds'] = [self.hparams.alpha_min, self.hparams.alpha_max]
        output_dict['v_bounds'] = [self.hparams.driving_vel_min, self.hparams.driving_vel_max]
        output_dict['omega_bounds'] = [self.hparams.steering_vel_max_neg, self.hparams.steering_vel_max]
        output_dict['wheels_vel_bounds'] = [self.hparams.w_max_neg, self.hparams.w_max]
        output_dict['vdot_bounds'] = [self.hparams.driving_acc_min, self.hparams.driving_acc_max]
        output_dict['omegadot_bounds'] = [self.hparams.steering_acc_max_neg, self.hparams.steering_acc_max]

        output_dict['robot_radius'] = self.hparams.rho_cbf
        output_dict['agent_radius'] = self.hparams.ds_cbf
        output_dict['gamma_area'] = self.hparams.gamma_area
        output_dict['gamma_agent'] = self.hparams.gamma_agent
        output_dict['frequency'] = self.hparams.generator_frequency
        output_dict['dt'] = self.hparams.dt
        output_dict['N_horizon'] = self.hparams.N_horizon
        output_dict['position_weight'] = self.hparams.p_weight
        output_dict['v_weight'] = self.hparams.v_weight
        output_dict['omega_weight'] = self.hparams.omega_weight
        output_dict['input_weight'] = self.hparams.u_weight
        output_dict['heading_weight'] = self.hparams.h_weight
        output_dict['terminal_factor_p'] = self.hparams.terminal_factor_p
        output_dict['terminal_factor_v'] = self.hparams.terminal_factor_v
        output_dict['b'] = self.hparams.b
        output_dict['base_radius'] = self.hparams.base_radius
        output_dict['wheel_radius'] = self.hparams.wheel_radius
        output_dict['wheel_separation'] = self.hparams.wheel_separation
        output_dict['laser_rel_pos'] = self.laser_relative_position.tolist()
        if self.hparams.perception in (Perception.CAMERA, Perception.BOTH):
            output_dict['camera_position'] = self.camera_pos_history
            output_dict['camera_horz_angle'] = self.camera_horz_angle_history

        # log the data in a .json file
        log_dir = self.hparams.log_dir
        filename = self.hparams.generator_file
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, filename)
        with open(log_path, 'w') as file:
            json.dump(output_dict, file)

    def update_control_input(self):
        if self.status == Status.MOVING:
            indexes = get_area_index(self.hparams.areas,
                                    self.hparams.a_coefs,
                                    self.hparams.b_coefs,
                                    self.hparams.c_coefs,
                                    self.state.get_state()[:2],
                                    self.hparams.rho_cbf)
            if self.next_area_idx in indexes:
                self.area_idx = self.next_area_idx
                self.next_area_idx, self.current_target_position = self.planner.get_next_step()
                if self.current_target_position is None:
                    self.current_target_position = self.target_position
            q_ref = np.zeros((self.nmpc_controller.nq, self.hparams.N_horizon+1))
            for k in range(self.hparams.N_horizon):
                q_ref[:self.hparams.y_idx + 1, k] = self.current_target_position
            u_ref = np.zeros((self.nmpc_controller.nu, self.hparams.N_horizon))
            q_ref[:self.hparams.y_idx + 1, self.hparams.N_horizon] = self.current_target_position
            # Compute the position and velocity error
            self.error = np.array([self.target_position[self.hparams.x_idx] - self.state.x, 
                                   self.target_position[self.hparams.y_idx] - self.state.y,
                                   0.0 - self.state.v,
                                   0.0 - self.state.omega])
            
            if norm(self.error) < self.hparams.nmpc_error_tol:
                control_input = np.zeros(self.nmpc_controller.nu)
                print("Stop state ###############################")
                print(self.state)
                print("##########################################")
                self.status = Status.READY
            else:
                try:
                    self.nmpc_controller.update(
                        self.state,
                        q_ref,
                        u_ref,
                        self.area_idx,
                        self.crowd_motion_prediction_stamped.crowd_motion_prediction.motion_predictions
                    )
                    control_input = self.nmpc_controller.get_control_input()
                except Exception as e:
                    rospy.logwarn("NMPC solver failed")
                    rospy.logwarn('{}'.format(e))
                    control_input = np.zeros(self.nmpc_controller.nu)
                    print("Failure state ############################")
                    print(self.state)
                    print("##########################################")  
        else:
            control_input = np.zeros(self.nmpc_controller.nu)

        return control_input

    def point_head(self, measurements = None):
            # Select the target to point: the closest agent, if any, or the target position
            if measurements is None or len(measurements) == 0 or self.hparams.perception == Perception.CAMERA:
                if norm(self.error) > self.hparams.pointing_error_tol:
                    point = self.current_target_position
                else:
                    return                
            else:
                sorted_measures, _ = sort_by_distance(np.array(measurements), self.state.get_state()[:2])
                point = sorted_measures[0]
            homo_point = np.append(np.append(point, self.camera_pose[2, 3]), 1)
            baseframe_point = np.matmul(self.base_pose, homo_point)
            point_head_msg = control_msgs.msg.PointHeadActionGoal()
            point_head_msg.goal.target.header.frame_id = self.base_footprint_frame
            point_head_msg.goal.target.point.x = baseframe_point[0]
            point_head_msg.goal.target.point.y = baseframe_point[1]
            point_head_msg.goal.target.point.z = baseframe_point[2]
            point_head_msg.goal.pointing_axis.x = 0.0
            point_head_msg.goal.pointing_axis.y = 0.0
            point_head_msg.goal.pointing_axis.z = 1.0
            point_head_msg.goal.pointing_frame = self.camera_frame

            point_head_msg.goal.min_duration = rospy.Duration(0.1)
            point_head_msg.goal.max_velocity = 0.5

            self.point_head_action_publisher.publish(point_head_msg)
            
    def run(self):
        rate = rospy.Rate(self.hparams.generator_frequency)

        if self.hparams.log:
            rospy.on_shutdown(self.log_values)

        while not(rospy.is_shutdown()):
            start_time = time.time()

            if self.status == Status.WAITING:
                self.laser_relative_position = self.get_laser_relative_position(rospy.Time())
                if self.update_state(rospy.Time()) and self.get_camera_pose(rospy.Time()) and self.laser_relative_position is not None:
                    self.init()
                    self.status = Status.READY
                    print("Initial state ****************************")
                    print(self.state)
                    print("******************************************")
                else:
                    rate.sleep()
                    continue

            if self.hparams.n_filters > 0 and self.crowd_motion_prediction_stamped_nonrt is None:
                rospy.logwarn("Missing crowd data")
                rate.sleep()
                continue

            with self.data_lock:
                if self.hparams.n_filters > 0:
                    self.crowd_motion_prediction_stamped = self.crowd_motion_prediction_stamped_nonrt
                    timestamp = self.crowd_motion_prediction_stamped.time
                else:
                    timestamp = rospy.Time()
                measurements = self.adapt_measurements_format(self.measurements_stamped_nonrt)
                wheels_vel = self.wheels_vel_nonrt
                if self.hparams.simulation:
                    agents_pos = self.agents_pos_nonrt

            self.update_state(rospy.Time())
            self.get_camera_pose(rospy.Time())
            self.point_head(measurements)
            # Generate control inputs (wheels accelerations)
            control_input = self.update_control_input()
            # Publish commands (robot pseudovelocities)
            v_cmd, omega_cmd = self.publish_command(control_input)
            
            # Update logged data
            if self.hparams.log:
                self.robot_state_history.append([self.state.x,
                                                 self.state.y,
                                                 self.state.theta,
                                                 self.state.v,
                                                 self.state.omega,
                                                 start_time])
                if self.hparams.perception in (Perception.CAMERA, Perception.BOTH):
                    self.camera_pos_history.append(self.camera_pose[:2, 3].tolist())
                    self.camera_horz_angle_history.append(self.camera_horz_angle)
                predicted_trajectory = np.zeros((self.hparams.N_horizon+1, 2))
                for i in range(self.hparams.N_horizon):
                    predicted_trajectory[i] = self.nmpc_controller.acados_ocp_solver.get(i,'x')[:2]
                predicted_trajectory[self.hparams.N_horizon] = \
                    self.nmpc_controller.acados_ocp_solver.get(self.hparams.N_horizon, 'x')[:2]
                self.robot_prediction_history.append(predicted_trajectory.tolist())
                self.wheels_vel_history.append([wheels_vel[self.hparams.r_wheel_idx],
                                                wheels_vel[self.hparams.l_wheel_idx],
                                                start_time])
                self.inputs_history.append([control_input[self.hparams.r_wheel_idx],
                                            control_input[self.hparams.l_wheel_idx],
                                            start_time])
                self.commands_history.append([v_cmd, omega_cmd, start_time])
                self.target_history.append([self.target_position[self.hparams.x_idx],
                                            self.target_position[self.hparams.y_idx],
                                            start_time])
                self.area_index_history.append(self.area_idx)
                self.target_viapoint_history.append(self.current_target_position.tolist())

                if self.hparams.n_filters > 0:
                    predicted_agent_trajectories = []
                    for motion_prediction in self.crowd_motion_prediction_stamped.crowd_motion_prediction.motion_predictions:
                        predicted_agent_trajectories.append([motion_prediction.position.x,
                                                             motion_prediction.position.y,
                                                             motion_prediction.velocity.x,
                                                             motion_prediction.velocity.y])
                    self.agents_prediction_history.append(predicted_agent_trajectories)

                if self.hparams.simulation:
                    self.agents_pos_history.append(agents_pos.tolist())
                end_time = time.time()        
                deltat = end_time - start_time
                self.time_history.append([deltat, start_time])
                if deltat > 1 / (2 * self.hparams.generator_frequency):
                    print(f"Iteration time {deltat} at instant {start_time}")

            rate.sleep()

def main():
    rospy.init_node('tiago_motion_generation', log_level=rospy.INFO)
    rospy.loginfo('TIAGo motion generation module [OK]')

    # Build and run motion generation manager
    motion_generation_manager = MotionGenerationManager()
    prof_filename = '/tmp/generator.prof'
    cProfile.runctx(
        'motion_generation_manager.run()',
        globals=globals(),
        locals=locals(),
        filename=prof_filename
    )
