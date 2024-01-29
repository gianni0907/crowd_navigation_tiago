import numpy as np
import os
import time
import json
import math
import rospy
import threading
import tf2_ros
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

from my_tiago_controller.utils import *
from my_tiago_controller.Hparams import *
from my_tiago_controller.Status import *
from my_tiago_controller.FSM import *

import sensor_msgs.msg
import my_tiago_msgs.srv
import my_tiago_msgs.msg

def z_rotation(angle, point2d):
    R = np.array([[math.cos(angle), - math.sin(angle), 0.0],
                  [math.sin(angle), math.cos(angle), 0.0],
                  [0.0, 0.0, 1.0]])
    point3d = np.array([point2d[0], point2d[1], 0.0])
    rotated_point2d = np.matmul(R, point3d)[:2]
    return rotated_point2d

def polar2relative(scan, angle_min, angle_incr):
    relative_laser_pos = Hparams.relative_laser_pos
    idx = scan[0]
    distance = scan[1]
    angle = angle_min + idx * angle_incr
    xy_relative = np.array([distance * math.cos(angle) + relative_laser_pos[0],
                            distance * math.sin(angle) + relative_laser_pos[1]])
    return xy_relative

def polar2absolute(scan, state, angle_min, angle_incr):
    xy_relative = polar2relative(scan, angle_min, angle_incr)
    xy_absolute = z_rotation(state.theta, xy_relative) + np.array([state.x, state.y])
    return xy_absolute

def moving_average(polar_scans):
    smoothed_scans = []
    window_size = 5

    for i, (idx, value) in enumerate(polar_scans):
        # Compute indices for the moving window
        start_idx = np.max([0, i - window_size // 2])
        end_idx = np.min([len(polar_scans), i + window_size // 2 + 1])

        # Extract the values within the moving window
        window_values = np.array(polar_scans[start_idx : end_idx])

        average_value = np.sum(window_values[:, 1]) / window_values.shape[0]
        smoothed_scans.append((idx, average_value))

    return smoothed_scans

def data_preprocessing(scans, tiago_state, range_min, angle_min, angle_incr):
    n_edges = Hparams.n_points
    vertexes = Hparams.vertexes
    normals = Hparams.normals
    polar_scans = []
    absolute_scans = []

    # Delete the first and last 20 laser scan ranges (wrong measurements?)
    offset = Hparams.offset
    scans = np.delete(scans, range(offset), 0)
    scans = np.delete(scans, range(len(scans) - offset, len(scans)), 0)

    for idx, value in enumerate(scans):
        outside = False
        if value != np.inf and value >= range_min:
            absolute_scan = polar2absolute((idx + offset, value), tiago_state, angle_min, angle_incr)
            for i in range(n_edges - 1):
                vertex = vertexes[i + 1]
                if np.dot(normals[i], absolute_scan - vertex) < 0.0:
                    outside = True
                    break
            if not outside:
                vertex = vertexes[0]
                if np.dot(normals[n_edges - 1], absolute_scan - vertex) < 0.0:
                    outside = True
            if not outside:
                polar_scans.append((idx + offset, value))
                absolute_scans.append(absolute_scan.tolist())

    return absolute_scans, polar_scans

def data_clustering(absolute_scans, polar_scans):
    if len(absolute_scans) != 0:
        k_means = DBSCAN(eps=0.3, min_samples=8)
        clusters = k_means.fit_predict(np.array(absolute_scans))
        dynamic_n_clusters = max(clusters) + 1
        if(min(clusters) == -1):
            print("Noisy samples")
        polar_core_points = np.zeros((dynamic_n_clusters, 2))
        
        for cluster_i in range(dynamic_n_clusters):
            cluster_scans = []
            for id, scan in zip(clusters, polar_scans):
                if id == cluster_i:
                    cluster_scans.append(scan)

            smoothed_scans = moving_average(cluster_scans)
            min_distance = np.inf
            for idx, scan in enumerate(smoothed_scans):            
                if scan[1] < min_distance:
                    min_distance = scan[1]
                    polar_core_points[cluster_i] = smoothed_scans[idx]

        polar_core_points = polar_core_points.tolist()        
        polar_core_points.sort(key = lambda x: x[1], reverse = False)
        polar_core_points = np.array(polar_core_points[0:Hparams.n_clusters])
    else:
        polar_core_points = np.array([])

    return polar_core_points

class CrowdPredictionManager:
    '''
    From the laser scans input predict the motion of the actors
    '''
    def __init__(self):
        self.data_lock = threading.Lock()

        # Set status
        # 2 scenarios:
        #   self.hparams.fake_sensing == True -> 3 possible status
        #       WAITING for the initial robot state
        #       READY to get actors trajectory
        #       MOVING when actors are moving
        #   self.hparams.fake_sensing == False -> 2 possible status
        #       WAITING for the initial robot state
        #       READY to read from laser scan topic and to move
        self.status = Status.WAITING # WAITING for the initial robot state

        self.robot_state = State(0.0, 0.0, 0.0, 0.0, 0.0)
        self.wheels_vel = np.zeros(2) # [w_r, w_l]
        self.hparams = Hparams()
        self.laser_scan = None
        self.n_actors = self.hparams.n_actors
        self.n_clusters = self.hparams.n_clusters
        self.actors_position = np.zeros((self.hparams.n_clusters, 2))

        self.N_horizon = self.hparams.N_horizon
        self.frequency = self.hparams.controller_frequency
        self.dt = self.hparams.dt

        # Set variables to store data  
        if self.hparams.log:
            self.kalman_infos = {}
            kalman_names = ['KF_{}'.format(i + 1) for i in range(self.n_clusters)]
            self.kalman_infos = {key: list() for key in kalman_names}
            self.time_history = []
            if not self.hparams.fake_sensing:
                self.scans_history = []
            self.actors_history = []
            self.robot_state_history = []

        # Setup reference frames:
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'

        # Setup TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Setup subscriber for joint_states topic
        state_topic = '/joint_states'
        rospy.Subscriber(
            state_topic,
            sensor_msgs.msg.JointState,
            self.joint_states_callback
        )

        # Setup subscriber to scan_raw topic
        scan_topic = '/scan_raw'
        rospy.Subscriber(
            scan_topic,
            sensor_msgs.msg.LaserScan,
            self.laser_scan_callback
        )

        # Setup publisher for crowd motion prediction:
        crowd_prediction_topic = 'crowd_motion_prediction'
        self.crowd_motion_prediction_publisher = rospy.Publisher(
            crowd_prediction_topic,
            my_tiago_msgs.msg.CrowdMotionPredictionStamped,
            queue_size=1
        )

        # Setup ROS Service to set actors trajectories:
        self.set_actors_trajectory_srv = rospy.Service(
            'SetActorsTrajectory',
            my_tiago_msgs.srv.SetActorsTrajectory,
            self.set_actors_trajectory_request
        )

    def joint_states_callback(self, msg):
        self.wheels_vel = np.array([msg.velocity[13], msg.velocity[12]])

    def laser_scan_callback(self, msg):
        self.data_lock.acquire()
        self.laser_scan = LaserScan.from_message(msg)
        self.data_lock.release()

    def set_from_tf_transform(self, transform):
        q = transform.transform.rotation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.robot_state.theta = theta
        self.robot_state.x = transform.transform.translation.x + self.hparams.b * math.cos(theta)
        self.robot_state.y = transform.transform.translation.y + self.hparams.b * math.sin(theta)

    def set_actors_trajectory_request(self, request):
        if not self.hparams.fake_sensing:
            rospy.loginfo("Cannot set synthetic trajectories, real sensing is active")
            return my_tiago_msgs.srv.SetActorsTrajectoryResponse(False) 
        else:
            if self.status == Status.WAITING:
                rospy.loginfo("Cannot set actors trajectory, robot is not READY")
                return my_tiago_msgs.srv.SetActorsTrajectoryResponse(False)            
            elif self.status == Status.READY:
                self.trajectories = CrowdMotionPrediction.from_message(request.trajectories)
                self.status = Status.MOVING
                self.current = 0
                self.trajectory_length = len(self.trajectories.motion_predictions[0].positions)
                rospy.loginfo("Actors trajectory successfully set")
                return my_tiago_msgs.srv.SetActorsTrajectoryResponse(True)
            else:
                rospy.loginfo("Cannot set actors trajectory, actors are already moving")
                return my_tiago_msgs.srv.SetActorsTrajectoryResponse(False)

    def update_actors_position(self):
            actors_position = np.zeros((self.n_clusters, 2))
            if self.hparams.fake_sensing:
                for i in range(self.n_clusters):
                            actors_position[i] = np.array(
                                [self.trajectories.motion_predictions[i].positions[self.current].x,
                                 self.trajectories.motion_predictions[i].positions[self.current].y]
                            )
                self.current += 1
                if self.current == self.trajectory_length:
                    self.current = 0
            else:
                angle_min = self.laser_scan.angle_min
                angle_increment = self.laser_scan.angle_increment
                range_min = self.laser_scan.range_min

                # Perform data preprocessing
                self.absolute_scans = []
                self.polar_scans = []
                self.absolute_scans, self.polar_scans = data_preprocessing(self.laser_scan.ranges,
                                                                           self.robot_state,
                                                                           range_min,
                                                                           angle_min,
                                                                           angle_increment)
                
                # Perform data clustering
                actors_polar_position = data_clustering(self.absolute_scans,
                                                        self.polar_scans)
                
                for (i, dist) in enumerate(actors_polar_position):
                    actors_position[i] = polar2absolute(actors_polar_position[i],
                                                        self.robot_state,
                                                        angle_min,
                                                        angle_increment)
                    
            self.data_lock.acquire()
            self.actors_position = actors_position
            self.data_lock.release()

    def update_state(self):
        try:
            # Update [x, y, theta]
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_footprint_frame, rospy.Time()
            )
            self.set_from_tf_transform(transform)
            # Update [v, omega]
            self.robot_state.v = self.hparams.wheel_radius * 0.5 * \
                (self.wheels_vel[self.hparams.r_wheel_idx] + self.wheels_vel[self.hparams.l_wheel_idx])
            
            self.robot_state.omega = (self.hparams.wheel_radius / self.hparams.wheel_separation) * \
                (self.wheels_vel[self.hparams.r_wheel_idx] - self.wheels_vel[self.hparams.l_wheel_idx])
            return True
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing current state")
            return False

    def propagate_state(self, state, N):
        predictions = [self.hparams.nullstate for _ in range(N)]
        time = 0
        dt = self.dt
        for i in range(N):
            time = dt * (i)
            predictions[i] = self.predict_next_state(state, time)

        return predictions

    def predict_next_state(self, state, dt):
        F = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        next_state = np.matmul(F, state)

        return next_state

    def log_values(self):
        output_dict = {}
        output_dict['kfs'] = self.kalman_infos
        output_dict['robot_states'] = self.robot_state_history
        output_dict['cpu_time'] = self.time_history
        output_dict['actors_position'] = self.actors_history
        if not self.hparams.fake_sensing:
            output_dict['laser_scans'] = self.scans_history
            output_dict['angle_min'] = self.laser_scan.angle_min
            output_dict['angle_max'] = self.laser_scan.angle_max
            output_dict['angle_inc'] = self.laser_scan.angle_increment
            output_dict['laser_offset'] = self.hparams.offset
            output_dict['range_min'] = self.laser_scan.range_min
            output_dict['range_max'] = self.laser_scan.range_max
            output_dict['laser_relative_pos'] = self.hparams.relative_laser_pos.tolist()
        
        # log the data in a .json file
        log_dir = '/tmp/crowd_navigation_tiago/data'
        filename = self.hparams.prediction_file
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, filename)
        with open(log_path, 'w') as file:
            json.dump(output_dict, file)


    def run(self):
        rate = rospy.Rate(self.frequency)
        if self.hparams.log:
            rospy.on_shutdown(self.log_values)

        if self.n_actors == 0:
            rospy.logwarn("No actors available")
            return
        
        if self.hparams.use_kalman:
            fsms = [FSM(self.hparams) for _ in range(self.n_clusters)]

        while not rospy.is_shutdown():
            start_time = time.time()
            
            if self.status == Status.WAITING:
                if self.update_state():
                    self.status = Status.READY
                    print("Initial state ****************************")
                    print(self.robot_state)
                    print("******************************************")
                else:
                    rate.sleep()
                    continue
            
            if self.hparams.fake_sensing and self.status == Status.READY:
                rospy.logwarn("Missing fake sensing info")
                rate.sleep()
                continue
            
            if self.laser_scan is None and not self.hparams.fake_sensing:
                rospy.logwarn("Missing laser scan info")
                rate.sleep()
                continue

            self.data_lock.acquire()
            self.update_state()
            self.data_lock.release()

            # Reset crowd_motion_prediction message
            crowd_motion_prediction = CrowdMotionPrediction()
            
            # Update the actors position (based on fake or real sensing)
            self.update_actors_position()
            
            # Saving data for plots
            if self.hparams.log:
                self.robot_state_history.append([
                    self.robot_state.x,
                    self.robot_state.y,
                    self.robot_state.theta,
                    self.robot_state.v,
                    self.robot_state.omega,
                    start_time
                ])
                self.actors_history.append(self.actors_position.tolist())
                if not self.hparams.fake_sensing:
                    self.scans_history.append(self.absolute_scans)

            if self.hparams.use_kalman:
                # Perform data association
                # predict next positions according to fsms
                next_predicted_positions = np.zeros((self.hparams.n_clusters, 2))
                associated_clusters = []

                for (i, fsm) in enumerate(fsms):
                    if fsm.next_state in (FSMStates.ACTIVE, FSMStates.HOLD):
                        predicted_state = self.propagate_state(fsm.current_estimate, 2)[1]
                    elif fsm.next_state is FSMStates.START:
                        predicted_state = fsm.current_estimate
                    else:
                        predicted_state = self.hparams.nullstate
                    next_predicted_positions[i] = predicted_state[:2]
                
                # compute pairwise distance between predicted positions and positions from clusters
                if self.actors_position.shape[0] > 0:
                    distances = cdist(next_predicted_positions, self.actors_position)
                else:
                    distances = None

                # match each fsm to the closest cluster centroid
                for (i, fsm) in enumerate(fsms):
                    fsm_state = fsm.next_state
                    
                    if self.hparams.log:
                        self.kalman_infos['KF_{}'.format(i + 1)].append([FSMStates.print(fsm.state),
                                                                         FSMStates.print(fsm_state),
                                                                         start_time])

                    # find the closest available cluster to the fsm's prediction
                    min_dist = np.inf
                    cluster = None
                    if distances is not None:
                        for j in range(distances.shape[1]):
                            if j in associated_clusters:
                                continue
                            distance = distances[i, j]
                            if distance < min_dist:
                                min_dist = distance
                                cluster = j
                    if cluster is not None:
                        measure = self.actors_position[cluster]
                        fsm.update(start_time, measure)
                        associated_clusters.append(cluster)
                    else:
                        measure = np.array([0.0, 0.0])
                        fsm.update(start_time, measure)

                    current_estimate = fsm.current_estimate
                    predictions = self.propagate_state(current_estimate, self.N_horizon)
                    predicted_positions = [Position(0.0, 0.0) for _ in range(self.N_horizon)]
                    predicted_velocities = [Velocity(0.0, 0.0) for _ in range(self.N_horizon)]
                    for j in range(self.N_horizon):
                        predicted_positions[j] = Position(predictions[j][0], predictions[j][1])
                        predicted_velocities[j] = Velocity(predictions[j][2], predictions[j][3])
                    
                    crowd_motion_prediction.append(
                            MotionPrediction(predicted_positions, predicted_velocities)
                    )
            else:
                for i in range(self.hparams.n_clusters):
                    if any(coord != 0.0 for coord in self.actors_position[i]):
                        current_state = np.array([self.actors_position[i, 0],
                                                  self.actors_position[i, 1],
                                                  0.0,
                                                  0.0])
                    else:
                        current_state = self.hparams.nullstate

                    predictions = self.propagate_state(current_state, self.N_horizon)
                    predicted_positions = [Position(0.0, 0.0) for _ in range(self.N_horizon)]
                    predicted_velocities = [Velocity(0.0, 0.0) for _ in range(self.N_horizon)]
                    for j in range(self.N_horizon):
                        predicted_positions[j] = Position(predictions[j][0], predictions[j][1])
                        predicted_velocities[j] = Velocity(predictions[j][2], predictions[j][3])
                
                    crowd_motion_prediction.append(
                        MotionPrediction(predicted_positions, predicted_velocities)
                    )

            crowd_motion_prediction_stamped = CrowdMotionPredictionStamped(rospy.Time.from_sec(start_time),
                                                                           'map',
                                                                           crowd_motion_prediction)
            crowd_motion_prediction_stamped_msg = CrowdMotionPredictionStamped.to_message(crowd_motion_prediction_stamped)
            self.crowd_motion_prediction_publisher.publish(crowd_motion_prediction_stamped_msg)
            
            if self.hparams.log:
                end_time = time.time()
                deltat = end_time - start_time
                self.time_history.append([deltat, start_time])

            rate.sleep()

def main():
    rospy.init_node('tiago_crowd_prediction', log_level=rospy.INFO)
    rospy.loginfo('TIAGo crowd prediction module [OK]')

    crowd_prediction_manager = CrowdPredictionManager()
    crowd_prediction_manager.run()